"""Resume from experiment results: aggregate plots + write paper + review."""
import os, sys, shutil, json
import os.path as osp

os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))

from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_icbinb_writeup import perform_writeup, gather_citations
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.llm import create_client

# Use the existing experiment dir
idea_dir = "experiments/2026-05-20_17-17-37_compositional_regularization_nn_attempt_0"
print(f"Resuming from: {idea_dir}")

# Step 1: Aggregate plots
print("\n=== Step 1: Aggregate plots ===")
aggregate_plots(base_folder=idea_dir, model="gpt-4o-2024-11-20")

# Clean up raw experiment results
experiment_results_dir = osp.join(idea_dir, "experiment_results")
if os.path.exists(experiment_results_dir):
    shutil.rmtree(experiment_results_dir)
print("Plots aggregated.")

# Step 2: Gather citations
print("\n=== Step 2: Gather citations ===")
citations_text = gather_citations(
    idea_dir,
    num_cite_rounds=20,
    small_model="gpt-4o-2024-11-20",
)
print("Citations gathered.")

# Step 3: Write paper (with retries)
print("\n=== Step 3: Write paper ===")
for attempt in range(3):
    print(f"Writeup attempt {attempt+1}/3")
    writeup_success = perform_writeup(
        base_folder=idea_dir,
        small_model="gpt-4o-2024-05-13",
        big_model="gpt-4o-2024-11-20",
        page_limit=4,
        citations_text=citations_text,
    )
    if writeup_success:
        print("Writeup succeeded!")
        break
else:
    print("Writeup failed after all retries.")

# Step 4: Review
print("\n=== Step 4: Review ===")
pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
if pdf_files:
    pdf_path = osp.join(idea_dir, pdf_files[0])
    print(f"Paper found: {pdf_path}")
    paper_content = load_paper(pdf_path)
    client, client_model = create_client("gpt-4o-2024-11-20")
    review_text = perform_review(paper_content, client_model, client)
    review_img = perform_imgs_cap_ref_review(client, client_model, pdf_path)
    with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
        f.write(json.dumps(review_text, indent=4))
    with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
        json.dump(review_img, f, indent=4)
    print("Review completed.")
else:
    print("No PDF found for review.")

print(f"\nDone! Check {idea_dir}/ for output files.")
