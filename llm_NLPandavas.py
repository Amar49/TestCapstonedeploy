from clinical_main import output_llm
import random
import gradio as gr
import clinical_main

pdf=[]
for x in clinical_main.pdfs:
    pdf.append(x.split(".")[0])
    
inputs = [
    gr.Dropdown(
                label="Clinical UOM",
                choices=pdf,
                value=lambda: random.choice(pdf)),
    gr.outputs.Textbox()
]


gr.Interface(fn=output_llm, inputs=inputs, outputs=["text"]).launch(server_name='0.0.0.0', server_port=8080)
