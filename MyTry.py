from EasyEdit.easyeditor import MultimodalEditor
from easyeditor import MENDMultimodalHparams
hparams = MENDMultimodalHparams.from_hparams('./hparams/MEND/minigpt4')

prompts = [
    "How many tennis balls are in the picture?",
    "What is the red food?"
]

targets = ["2", "tomatoes",]

image = [
    "../dataset/val2014/COCO_val2014_000000451435.jpg",
    "../dataset/val2014/COCO_val2014_000000189446.jpg"
]


editor = MultimodalEditor.from_hparams(hparams)

locality_inputs = {
    'text': {
        'prompt': [
            "nq question: what purpose did seasonal monsoon winds have on trade"
          ],
        'ground_truth': [
            "enabled European empire expansion into the Americas and trade  \
            routes to become established across the Atlantic and Pacific oceans"
          ]
    },
    'vision': {
        'prompt': ["What sport can you use this for?"],
        'ground_truth': ["riding"],
        'image': ["../dataset/val2014/COCO_val2014_000000297147.jpg"],
    }
}
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    target_new=targets,
    image=image,
    locality_inputs=locality_inputs,
    keep_original_weight=False
)
## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model
print(metrics)
# print(edited_model)