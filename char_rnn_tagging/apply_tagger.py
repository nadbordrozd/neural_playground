import argparse
import os
from glob import glob
import matplotlib
import numpy as np
from keras.models import load_model
from train import generate_batches



def prediction_to_html(text, predictions, labels, cmap='Reds'):
    cmap = matplotlib.cm.get_cmap(cmap)
    html_chars = []
    for c, p, l in zip(text, predictions, labels):
        if c == '\n':
            html_chars.append('<br>')
        else:
            r, g, b, a = cmap(p)
            r, g, b = int(256*r), int(256*g), int(256*b)
            if l:
                c = '<font face="Times New Roman" size="5">%s</font>' % c
            else:
                c = '<font face="monospace" size="3">%s</font>' % c
            html_chars.append('<span style="background-color:rgb(%s, %s, %s); color:black;">%s</span>' % (r, g, b, c))
    tot_html = "".join(html_chars)
    return tot_html


def get_batches_and_text(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, n):
    """first yields n batches, then yields a list of texts + all the labels"""
    gen = generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, True)
    texts = []
    labels = []
    for i in range(n):
        X, y, txt = gen.next()
        texts.append(txt)
        labels.append(y.reshape((batch_size, sample_len)))
        yield (X, y)
    yield ["".join(parts) for parts in zip(*texts)], np.hstack(labels)


def main(model_path, output_path, dir_a, dir_b, min_jump_a, max_jump_a, min_jump_b, max_jump_b,
         steps):
    model = load_model(model_path)
    fa = glob(os.path.join(dir_a, "*"))
    fb = glob(os.path.join(dir_b, "*"))
    juma = [min_jump_a, max_jump_a]
    jumb = [min_jump_b, max_jump_b]
    batch_size, seq_len, n_chars = model.input_shape
    gen = get_batches_and_text(fa, juma, fb, jumb, batch_size, seq_len, steps + 1)
    predictions = model.predict_generator(gen, steps=steps, max_q_size=1)
    texts, labels = gen.next()
    try:
        os.makedirs(output_path)
    except os.error:
        pass
    for i in range(batch_size):
        preds = np.vstack([predictions[j::batch_size, :].ravel() for j in range(batch_size)])
        path = os.path.join(output_path, 'part_' + str(i).zfill(5) + ".html")
        print 'doing %s' % i, path
        with open(path, "wb") as f:
            f.write(prediction_to_html(texts[i], preds[i], labels[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with a trained tagger")
    parser.add_argument("model_path", help="path to trained model")
    parser.add_argument("dir_a", help="directory with first set of input files (it should contain "
                                      "'test' subdirectory")
    parser.add_argument("dir_b", help="directory with the second set of input files (it shouold "
                                      "contain 'test' subdirectory)")
    parser.add_argument("--min_jump_a", type=int, default=20, help="snippets from source A will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_a", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--min_jump_b", type=int, default=20, help="snippets from source B will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_b", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--steps", type=int, default=50, help="how many batches to predict")
    args = parser.parse_args()

    main(args.model_path, args.output_path, args.dir_a, args.dir_b, args.min_jump_a,
         args.max_jump_a, args.min_jump_b, args.max_jump_b, args.steps)




