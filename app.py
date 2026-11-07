"""Build the QuickBite interface and score orders through predict.py."""

import gradio as gr

from predict import (
    CATEGORY_RULES,
    FEATURE_NAMES,
    NUMERIC_RULES,
    predict_order,
)


def assess_order(*values):
    """Validate submitted details and display the prediction."""
    try:
        order = dict(zip(FEATURE_NAMES, values, strict=True))
        result = predict_order(order)
    except ValueError as exc:
        return f"Input error: {exc}", "", ""

    action = (
        "Offer the customer a $4 voucher."
        if result["offer_voucher"]
        else "No voucher is recommended."
    )

    return (
        f"{action} "
        f"(Decision threshold: {result['threshold']:.0%} late risk.)",
        f"{result['late_probability']:.1%}",
        f"{result['on_time_probability']:.1%}",
    )


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown(
        "# QuickBite Late Order Check\n"
        "Check an order using the available order and courier details. "
        "Enter the proposed courier's vehicle and experience. "
        "The result supports, but does not replace, dispatcher judgment."
    )

    inputs = []

    for feature in FEATURE_NAMES:
        if feature in CATEGORY_RULES:
            choices = list(CATEGORY_RULES[feature])
            component = gr.Dropdown(
                choices=choices,
                label=feature,
                value=choices[0],
            )
        else:
            minimum, maximum, whole = NUMERIC_RULES[feature]
            component = gr.Number(
                label=f"{feature} ({minimum:g} to {maximum:g})",
                value=minimum,
            )

        inputs.append(component)

    submit = gr.Button("Check order", variant="primary")

    action_output = gr.Textbox(
        label="Voucher decision or input error",
        interactive=False,
    )

    with gr.Row():
        late_output = gr.Textbox(
            label="Chance of late delivery",
            interactive=False,
        )
        on_time_output = gr.Textbox(
            label="Chance of on-time delivery",
            interactive=False,
        )

    submit.click(
        fn=assess_order,
        inputs=inputs,
        outputs=[action_output, late_output, on_time_output],
    )


if __name__ == "__main__":
    demo.launch()