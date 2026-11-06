"""QuickBite dispatcher Gradio interface; all scoring goes through predict.py."""

import gradio as gr

from predict import (
    CATEGORY_RULES,
    FEATURES,
    NUMERIC_RULES,
    predict_order,
)


def assess_order(*values):
    """Convert submitted form values to an order and show a plain-language result."""
    try:
        result = predict_order(dict(zip(FEATURES, values, strict=True)))
    except ValueError as exc:
        return "Input error: " + str(exc), None, None, ""
    action = (
        "Offer the customer a $4 voucher."
        if result["offer_voucher"]
        else "No voucher is recommended."
    )
    return (
        f"{action} (Decision threshold: {result['threshold']:.0%} late risk.)",
        f"{result['late_probability']:.1%}",
        f"{result['on_time_probability']:.1%}",
        " ".join(result["warnings"]),
    )


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown(
        "# 🚚 QuickBite Late Order Check\n"
        "Check an order **before a courier is assigned**. "
        "The result supports, but does not replace, dispatcher judgment."
    )
    inputs = []
    for feature in FEATURES:
        if feature in CATEGORY_RULES:
            choices = list(CATEGORY_RULES[feature])
            component = gr.Dropdown(choices=choices, label=feature, value=choices[0])
        else:
            minimum, maximum, whole = NUMERIC_RULES[feature]
            component = gr.Number(
                label=f"{feature} ({minimum:g}–{maximum:g})",
                precision=0 if whole else None,
                value=None if feature == "prep_minutes" else minimum,
            )
        inputs.append(component)
    submit = gr.Button("Check order", variant="primary")
    action_output = gr.Textbox(label="Dispatch action", interactive=False)
    with gr.Row():
        late_output = gr.Textbox(label="Chance of late delivery", interactive=False)
        on_time_output = gr.Textbox(
            label="Chance of on-time delivery", interactive=False
        )
    warning_output = gr.Textbox(label="Data warning", interactive=False)
    submit.click(
        fn=assess_order,
        inputs=inputs,
        outputs=[action_output, late_output, on_time_output, warning_output],
    )

if __name__ == "__main__":
    demo.launch()
