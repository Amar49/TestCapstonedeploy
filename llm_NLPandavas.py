from clinical_main import output_llm
import random
import gradio as gr
import qa_clinical

pdf=[]
for x in qa_clinical.pdfs:
    pdf.append(x.split(".")[0])
    
inputs = [
    gr.Dropdown(
                label="Clinical UOM",
                choices=pdf,
                value=lambda: random.choice(pdf)),
    gr.outputs.Textbox()
]


gr.Interface(fn=output_llm, inputs=inputs, outputs=["text"]).launch(share=True)