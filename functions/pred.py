def remove_chaklas(text):
    chaklas = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']
    for chakla in chaklas:
        text = text.replace(chakla, '')
    return text

def remove_hamza(text):
    text = text.replace('ء', '')
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')
    text = text.replace('ٱ', 'ا')
    text = text.replace('ؤ', 'و')
    text = text.replace('ئ', 'ى')
    return text

def predict_class(pipe, comment_content):
    probabilities = pipe(comment_content)[0]
    # Find the entry with the highest score
    return max(probabilities, key=lambda p: p["score"])["label"]

def class_proba(pipe, comment_content):
    probabilities = pipe(comment_content)[0]
    # Create a dictionary mapping labels to scores
    return {proba['label']: proba['score'] for proba in probabilities}

def class_proba_batch(pipe, texts, batch_size=64):
    outputs = pipe(
        texts,
        batch_size=batch_size,
        truncation=True
    )

    return [
        {p["label"]: p["score"] for p in output}
        for output in outputs
    ]