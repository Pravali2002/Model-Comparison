import os
import json
import time
import random
import boto3
from dotenv import load_dotenv
from langsmith import Client

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
ANNOTATION_FILE = os.path.join(DATASET_DIR, "_annotations.coco.json")

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

# ------------------- Bedrock invocation -------------------
def invoke_model(model_id, image_path):
    retries = 3
    for attempt in range(retries):
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
        except Exception as e:
            print(f"❌ Could not open image {image_path}: {e}")
            return None

        instruction_text = (
            "Classify the strawberry image into one of these categories: "
            "Gray_Mold, Powdery_Mildew, Anthracnose, Missing_calyx, "
            "Uneven_Ripening, Unripe_Strawberry, Good_Quality, Fasciated_Strawberry. "
            "Return ONLY the category name."
        )

        try:
            response = bedrock.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"text": instruction_text},
                        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
                    ]
                }],
                inferenceConfig={"maxTokens": 50, "temperature": 0.0},
            )
            return response["output"]["message"]["content"][0].get("text", "").strip()
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt + random.uniform(0.2, 0.5)
                print(f"⚠️ Throttled by {model_id}, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            print(f"❌ Error invoking {model_id}: {e}")
            return None
    return None

# ------------------- LLM-as-a-Judge -------------------
def judge_similarity(expected: str, predicted: str, judge_model="amazon.nova-lite-v1:0"):
    """Use a small LLM to judge if the predicted and expected categories match semantically."""
    if not predicted or not expected:
        return False

    prompt = (
        f"You are an expert evaluator for strawberry disease classification.\n"
        f"Expected label: {expected}\n"
        f"Predicted label: {predicted}\n\n"
        "Do these refer to the same or equivalent category? Reply with only 'Yes' or 'No'."
    )

    try:
        response = bedrock.converse(
            modelId=judge_model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0.0},
        )
        output = response["output"]["message"]["content"][0].get("text", "").strip().lower()
        return "yes" in output
    except Exception as e:
        print(f"⚠️ Judge model failed: {e}")
        return False

# ------------------- LangSmith dataset -------------------
run_name = f"bedrock_model_comparison_{int(time.time())}"
print(f"\n📦 Creating LangSmith dataset: {run_name}")

try:
    client.create_dataset(dataset_name=run_name)
except Exception:
    print("⚠️ Dataset already exists, reusing...")

# ------------------- Evaluation loop -------------------
results = {"pixtral": {"correct": 0, "total": 0}, "nova": {"correct": 0, "total": 0}}
rows_logged = 0

for img in data["images"]:
    image_filename = img["file_name"]
    image_path = os.path.join(DATASET_DIR, image_filename)
    related_anns = [a for a in annotations if a["image_id"] == img["id"]]
    if not related_anns:
        continue

    label = categories.get(related_anns[0]["category_id"], "Unknown")
    print(f"\n🖼️ Processing {image_filename} | Expected: {label}")

    print("  Invoking Pixtral...")
    pixtral_raw = invoke_model(models["pixtral"], image_path)
    time.sleep(1.5)

    print("  Invoking Nova...")
    nova_raw = invoke_model(models["nova"], image_path)

    # LLM-based accuracy judgment
    pixtral_correct = judge_similarity(label, pixtral_raw or "")
    nova_correct = judge_similarity(label, nova_raw or "")

    results["pixtral"]["total"] += 1
    results["nova"]["total"] += 1
    if pixtral_correct:
        results["pixtral"]["correct"] += 1
        print(f"✅ Pixtral: {pixtral_raw}")
    else:
        print(f"❌ Pixtral: {pixtral_raw}")

    if nova_correct:
        results["nova"]["correct"] += 1
        print(f"✅ Nova: {nova_raw}")
    else:
        print(f"❌ Nova: {nova_raw}")

    pixtral_acc = (results["pixtral"]["correct"] / results["pixtral"]["total"]) * 100
    nova_acc = (results["nova"]["correct"] / results["nova"]["total"]) * 100

    client.create_example(
        dataset_name=run_name,
        inputs={"image_file": image_filename},
        outputs={
            "expected_category": label,
            "pixtral_pred": pixtral_raw or "Empty",
            "pixtral_accuracy_running_pct": f"{pixtral_acc:.2f}",
            "nova_pred": nova_raw or "Empty",
            "nova_accuracy_running_pct": f"{nova_acc:.2f}",
        },
    )
    rows_logged += 1

# ------------------- Final summary -------------------
print("\n📊 Model Comparison Summary:")
for model, stats in results.items():
    acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] else 0
    print(f"  {model.capitalize()} Accuracy: {acc:.2f}% ({stats['correct']}/{stats['total']})")

client.create_example(
    dataset_name=run_name,
    inputs={"summary": "overall accuracy"},
    outputs={
        "pixtral_accuracy_pct": f"{(results['pixtral']['correct'] / results['pixtral']['total'] * 100):.2f}" if results["pixtral"]["total"] > 0 else "N/A",
        "nova_accuracy_pct": f"{(results['nova']['correct'] / results['nova']['total'] * 100):.2f}" if results["nova"]["total"] > 0 else "N/A",
        "rows_logged": rows_logged,
    },
)

print(f"\n✅ Done — results logged to LangSmith dataset: {run_name}")
print("🔗 View it at: https://smith.langchain.com/datasets")
