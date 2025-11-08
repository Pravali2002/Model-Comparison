import os
import json
import time
import boto3
from dotenv import load_dotenv
from langsmith import Client
import re
from collections import defaultdict

# ------------------- Load environment variables -------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ------------------- Initialize Clients -------------------
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

client = Client(api_key=LANGCHAIN_API_KEY)

# ------------------- Dataset Paths -------------------
DATASET_DIR = "/Users/pravalika/model_comparison/data/train"
ANNOTATION_FILE = "/Users/pravalika/model_comparison/data/train/_annotations.coco.json"

# ------------------- Load Dataset -------------------
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in data["categories"]}
annotations = data.get("annotations", [])
images = {img["id"]: img["file_name"] for img in data["images"]}

# ------------------- Models to Compare -------------------
models = {
    "pixtral": "us.mistral.pixtral-large-2502-v1:0",
    "nova": "amazon.nova-lite-v1:0"
}

# ------------------- Helper Functions -------------------
def invoke_model(model_id, image_path):
    """Invoke model to classify disease category."""
    retries = 3
    for attempt in range(retries):
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            response = bedrock.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"text": "Classify the strawberry disease category from this image. Return only the disease name."},
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
                    ]
                }],
                inferenceConfig={"maxTokens": 100, "temperature": 0.2},
            )

            return response["output"]["message"]["content"][0].get("text", "").strip()

        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"⚠️ Throttled by {model_id}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"❌ Error invoking {model_id}: {e}")
            return None

def clean_prediction(text):
    """Extract clean disease name from model output."""
    if not text:
        return "Unknown"
    # Extract bolded text if present
    match = re.search(r"\*\*(.*?)\*\*", text)
    if match:
        return match.group(1).strip()
    # Remove common filler words
    text = re.sub(r"(The strawberries.*affected by|This appears to be|disease is|likely|looks like|seems like|appears to be)", "", text, flags=re.I)
    return text.strip().replace(".", "").title()

# ------------------- Create LangSmith Dataset -------------------
run_name = f"bedrock_model_comparison_{int(time.time())}"
print(f"\n📦 Creating LangSmith dataset: {run_name}")

try:
    dataset = client.create_dataset(dataset_name=run_name)
except Exception:
    dataset = client.read_dataset(dataset_name=run_name)
    print("⚠️ Dataset already exists. Reusing it.")

# ------------------- Evaluation -------------------
results = defaultdict(lambda: {"correct": 0, "total": 0})

for img in data["images"]:
    image_filename = img["file_name"]
    image_path = os.path.join(DATASET_DIR, image_filename)

    related_anns = [a for a in annotations if a["image_id"] == img["id"]]
    if not related_anns:
        continue

    category_id = related_anns[0]["category_id"]
    label = categories.get(category_id, "Unknown")

    print(f"\n🖼️ Processing {image_filename} (label: {label})")
    print("  Invoking Pixtral...")
    pixtral_output = invoke_model(models["pixtral"], image_path)
    time.sleep(2)
    print("  Invoking Nova...")
    nova_output = invoke_model(models["nova"], image_path)

    pixtral_pred = clean_prediction(pixtral_output)
    nova_pred = clean_prediction(nova_output)

    # Update metrics
    results["pixtral"]["total"] += 1
    results["nova"]["total"] += 1
    if pixtral_pred.lower() == label.lower():
        results["pixtral"]["correct"] += 1
    if nova_pred.lower() == label.lower():
        results["nova"]["correct"] += 1

    # Log to LangSmith
    client.create_example(
        dataset_name=run_name,
        inputs={"image_file": image_filename},
        outputs={
            "expected_category": label,
            "pixtral_pred": pixtral_pred or "Error/empty",
            "nova_pred": nova_pred or "Error/empty",
        },
    )

# ------------------- Accuracy Summary -------------------
print("\n📊 Model Comparison Summary:")
for model_name, stats in results.items():
    acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    print(f"  {model_name.capitalize()} Accuracy: {acc:.2f}% ({stats['correct']}/{stats['total']})")

# Log summary row to LangSmith
client.create_example(
    dataset_name=run_name,
    inputs={"summary": "Overall model comparison results"},
    outputs={
        "pixtral_accuracy": f"{(results['pixtral']['correct'] / results['pixtral']['total'])*100:.2f}%" if results["pixtral"]["total"] > 0 else "N/A",
        "nova_accuracy": f"{(results['nova']['correct'] / results['nova']['total'])*100:.2f}%" if results["nova"]["total"] > 0 else "N/A",
        "pixtral_correct": results["pixtral"]["correct"],
        "nova_correct": results["nova"]["correct"],
        "pixtral_total": results["pixtral"]["total"],
        "nova_total": results["nova"]["total"],
    },
)

print(f"\n✅ All results logged to LangSmith dataset: {run_name}")
print("🔗 View it at: https://smith.langchain.com/datasets")
