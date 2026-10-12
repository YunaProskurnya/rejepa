"""Small human-readable companion files for a full linear-probe JSON report."""

import csv
import json


def summary_paths(output):
    return [output.with_suffix(suffix) for suffix in ('.best.json', '.best.md', '.trials.csv')]


def write_probe_summary(output, metrics, context):
    best_json, best_md, trials_csv = summary_paths(output)
    best = metrics['best']
    best_json.write_text(json.dumps(dict(context=context, best=best), indent=2), encoding='utf-8')
    best_md.write_text(
        '# Best linear probe\n\n'
        f"- Representation: `{best['representation']}` ({best['feature_dim']} features)\n"
        f"- Head: `{best['head']}`\n"
        f"- Reference LR: {best['reference_lr']}; actual LR: {best['lr']}\n"
        f"- Weight decay: {best['weight_decay']}\n"
        f"- Epochs: {context['epochs']}; head batch: {context['batch_size']}\n"
        f"- Variants: {len(metrics['trials'])}; selected by val top-1, then val loss\n\n"
        '| Split | Top-1 (%) | Cross-entropy | Images |\n'
        '| --- | ---: | ---: | ---: |\n'
        f"| Train | {best['train_top1']:.4f} | {best['train_loss']:.6f} | {best['train_images']} |\n"
        f"| Val | {best['val_top1']:.4f} | {best['val_loss']:.6f} | {best['val_images']} |\n\n"
        f"Full report: `{output.name}`\n", encoding='utf-8')
    with trials_csv.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(best))
        writer.writeheader()
        writer.writerows(metrics['trials'])
    return [best_json, best_md, trials_csv]
