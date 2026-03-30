"""
=============================================================================
Gradio User Interface
=============================================================================
Provides:
  Tab 1: Upload receipt → extract → view structured JSON
  Tab 2: Ask questions about stored receipts
  Tab 3: View all stored receipts in a table
=============================================================================
"""

import gradio as gr
import requests
import json
import pandas as pd
from PIL import Image
import io

API_BASE = "http://localhost:8000"


def extract_receipt(image):
    if image is None:
        return {}, "Please upload a receipt image."
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    try:
        resp = requests.post(
            f"{API_BASE}/extract",
            files={"file": ("receipt.jpg", buf, "image/jpeg")},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        summary = (
            f"✓ Extracted successfully!\n"
            f"  Vendor    : {data.get('vendor','—')}\n"
            f"  Date      : {data.get('date','—')}\n"
            f"  Total     : {data.get('total','—')}\n"
            f"  Receipt ID: {data.get('receipt_id','—')}\n"
            f"  DB ID     : {data.get('receipt_db_id','—')}"
        )
        return data, summary
    except Exception as e:
        return {}, f"Error: {str(e)}"


def ask_question(question, receipt_id):
    if not question.strip():
        return "Please enter a question."
    payload = {"question": question}
    if receipt_id.strip():
        payload["receipt_id"] = receipt_id.strip()
    try:
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Answer: {data['answer']}\n\n"
            f"Sources: {', '.join(data.get('source_receipts', []))}\n"
            f"Reasoning: {data.get('reasoning','')}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


def load_all_receipts():
    try:
        resp = requests.get(f"{API_BASE}/receipts", timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return pd.DataFrame(columns=["id","vendor","date","total","receipt_id","created_at"])
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


with gr.Blocks(title="Receipt Intelligence System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧾 Receipt Information Extraction & QA System")
    gr.Markdown("Fine-tuned LayoutLMv3 + LLM-based Question Answering")

    with gr.Tab("📤 Extract Receipt"):
        gr.Markdown("Upload a receipt image to extract structured information.")
        with gr.Row():
            with gr.Column():
                img_input  = gr.Image(type="pil", label="Receipt Image")
                ext_btn    = gr.Button("Extract", variant="primary")
            with gr.Column():
                ext_json   = gr.JSON(label="Extracted Data")
                ext_status = gr.Textbox(label="Status", lines=7)
        ext_btn.click(extract_receipt, inputs=[img_input], outputs=[ext_json, ext_status])

    with gr.Tab("❓ Ask Questions"):
        gr.Markdown("Ask natural language questions about your receipts.")
        q_input    = gr.Textbox(label="Your Question",
                                placeholder="What is the total amount?")
        rid_input  = gr.Textbox(label="Receipt DB ID (optional — leave blank to search all)",
                                placeholder="e.g. 3f2a1b…")
        q_btn      = gr.Button("Ask", variant="primary")
        q_output   = gr.Textbox(label="Answer", lines=6)
        q_btn.click(ask_question, inputs=[q_input, rid_input], outputs=[q_output])

        gr.Examples(
            examples=[
                ["What is the total amount?", ""],
                ["What is the date of the receipt?", ""],
                ["Which vendor issued this receipt?", ""],
                ["What is the receipt ID?", ""],
            ],
            inputs=[q_input, rid_input],
        )

    with gr.Tab("📋 All Receipts"):
        gr.Markdown("View all extracted receipts stored in the database.")
        refresh_btn = gr.Button("Refresh", variant="secondary")
        receipts_df = gr.DataFrame(label="Stored Receipts")
        refresh_btn.click(load_all_receipts, outputs=[receipts_df])
        demo.load(load_all_receipts, outputs=[receipts_df])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
