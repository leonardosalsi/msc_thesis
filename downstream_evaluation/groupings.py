import json
import os
from enum import Enum

from utils.model_definitions import TASK_DEFINITIONS, MODELS

PRETRAINED_MODEL_DIR = '/shared/pretrained_models'
BENCHMARK_DIR = '/shared/data/downstream'
UTR_CLASS_DIR = '/shared/data/class_5utr'

class DATATYPE(Enum):
    TRAINING_CURVES = 1
    BENCHMARK = 2
    UTR_CLASS = 3

def get_model_alias_for_downstream(model_name):
    for m in MODELS:
        if m['name'] == model_name:
            tokenizer = 'n.o. 6-mer, ' if m['tokenizer'] == 'default' else 'overlapping, '
            context = '1kb' if m['context'] == '1' else '2kb'
            dataset = 'multi-species, ' if m['dataset'] == 'multi_species' else 'logan, '
            ewc = f'ewc-lambda={m["ewc_lambda"]}, ' if m['ewc'] else ''
            pca = f'pca-dim={m["pca_dim"]}:{m["pca_embed"]}, ' if m['pca'] else ''
            sharon = f'sharon={m["sharon"]}, ' if m['sharon'] else ''
            gc = f'gc={m["gc"]}, ' if m['gc'] else ''
            return f'{tokenizer}{context}{dataset}{ewc}{pca}{sharon}{gc}'.rstrip(', ')
    return None

def get_model_entry(model_name):
    for m in MODELS:
        if m['name'] == model_name:
            return m
    return None

def get_task_alias(task_name):
    for task in TASK_DEFINITIONS:
        if task['name'] == task_name:
            return task['alias']
    return None

def check_task_result_existence(name, baseline=False):
    benchmark_folder = os.path.join(BENCHMARK_DIR, name, 'checkpoint-6B')
    try:
        files = os.listdir(benchmark_folder)
    except FileNotFoundError:
        print(f"Folder {benchmark_folder} is not available.")
        return
    for t in TASK_DEFINITIONS:
        if baseline and 'gb_' in t['name']:
            continue
        if not f"{t['name']}.json" in files:
            print(f"Results of {t['name']} for {name} is not available.")

def get_trainer_state(name):
    trainer_state_path = os.path.join(PRETRAINED_MODEL_DIR, name, 'checkpoint-12000', 'trainer_state.json')
    if not os.path.exists(trainer_state_path):
        print(f"Trainer state for {name} is not available.")
        return None
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    return trainer_state

def get_benchmark_data(name, baseline=False):
    benchmark_folder = os.path.join(BENCHMARK_DIR, name, 'checkpoint-6B')
    if not os.path.exists(benchmark_folder):
        print(f"Benchmark results for {name} is not available.")
        return None
    files = os.listdir(benchmark_folder)
    if baseline:
        files = [s for s in files if not s.startswith("gb_") and not s.startswith("utr5")]
    file_paths = [os.path.join(benchmark_folder, f) for f in files]
    return file_paths

def get_utr_class_data(name, baseline=False):
    utr_class_folder = os.path.join(UTR_CLASS_DIR, name, 'checkpoint-6B')
    if not os.path.exists(utr_class_folder):
        print(f"Benchmark results for {name} is not available.")
        return None
    files = os.listdir(utr_class_folder)
    file_paths = [os.path.join(utr_class_folder, f) for f in files]
    return file_paths

def check_model_existence():
    for m in MODELS:
        model_path = os.path.join(PRETRAINED_MODEL_DIR, m['name'])
        if not os.path.exists(model_path):
            print(f"Model {m['name']} is not available.")
        else:
            print(f"✅ Model {m['name']} is available.")

def _collect_benchmark_data(group, baseline=False):
    data = {}
    for g in group:
        check_task_result_existence(g, baseline)
        benchmark_data = get_benchmark_data(g, baseline)

        if benchmark_data is not None:
            data[g] = benchmark_data

    return data

def _collect_utr_class_data(group, baseline=False):
    data = {}
    for g in group:
        benchmark_data = get_utr_class_data(g, baseline)

        if benchmark_data is not None:
            data[g] = benchmark_data

    return data

def _collect_training_data(group):
    data = {}
    for g in group:
        trainer_state = get_trainer_state(g)
        if trainer_state is not None:
            data[g] = trainer_state

    return data

def get_for_baseline_compare(type: DATATYPE):
    group = [
        'default_multi_species_no_cont',
        'default_multi_species_no_cont_100',
        'default_multi_species_no_cont_250',
        'default_multi_species_no_cont_500',
        'default_multi_species_no_cont_500_tg',
        'default_multi_species_no_cont_500_human'
    ]
    filename = 'compare_literature'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group, True), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_tokenization_compare(type: DATATYPE):
    group = [
        'default_multi_species',
        'overlap_multi_species'
    ]
    filename = 'compare_tokenization'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

MODEL_DICT = {
    'default_multi_species_no_cont': 'NT-50M (no continual)',
    'default_multi_species_no_cont_100': 'NT-100M (no continual)',
    'default_multi_species_no_cont_250': 'NT-250M (no continual)',
    'default_multi_species_no_cont_500': 'NT-500M (no continual)',
    'default_multi_species': 'NT-50M (no overlap, multispecies)',
    'default_multi_species_2kb': 'NT-50M (no overlap, multispecies, 2k ctx.)',
    'overlap_multi_species': 'NT-50M (overlap, multispecies)',
    'overlap_multi_species_2kb': 'NT-50M (overlap, multispecies, 2k ctx.)',
    'overlap_logan_no_ewc': 'NT-50M (overlap, logan, no EWC)',
    'overlap_logan_ewc_0_5': 'NT-50M (overlap, logan, EWC 0.5)',
    'overlap_logan_ewc_1': 'NT-50M (overlap, logan, EWC 1)',
    'overlap_logan_ewc_2': 'NT-50M (overlap, logan, EWC 2)',
    'overlap_logan_ewc_5': 'NT-50M (overlap, logan, EWC 5)',
    'overlap_logan_ewc_10': 'NT-50M (overlap, logan, EWC 10)',
    'overlap_logan_ewc_25': 'NT-50M (overlap, logan, EWC 25)',
    'default_multi_species_sh_gc': 'NT-50M (no overlap, multispecies, GC & Shannon)',
    'default_multi_species_2kb_sh_gc': 'NT-50M (no overlap, multispecies, GC & Shannon, 2k ctx.)',
    'overlap_multi_species_sh_gc': 'NT-50M (overlap, multispecies, GC & Shannon)',
    'overlap_multi_species_2kb_sh_gc': 'NT-50M (overlap, multispecies, GC & Shannon, 2k ctx.)',
    'overlap_multi_species_pca_cls_256': 'NT-50M (overlap, multispecies, contrastive CLS)',
    'overlap_multi_species_pca_mean_256': 'NT-50M (overlap, multispecies, contrastive mean-pool)',
    'default_logan_no_ewc': 'NT-50M (no overlap, logan, no EWC)',
    'default_logan_ewc_0_5': 'NT-50M (no overlap, logan, EWC 0.5)',
    'default_logan_ewc_1': 'NT-50M (no overlap, logan, EWC 1)',
    'default_logan_ewc_2': 'NT-50M (no overlap, logan, EWC 2)',
    'default_logan_ewc_5': 'NT-50M (no overlap, logan, EWC 5)',
    'default_logan_ewc_10': 'NT-50M (no overlap, logan, EWC 10)',
    'default_logan_ewc_25': 'NT-50M (no overlap, logan, EWC 25)',
    'default_multi_species_pca_cls_256': 'NT-50M (no overlap, multispecies, contrastive CLS)',
    'default_multi_species_pca_mean_256': 'NT-50M (no overlap, multispecies, contrastive mean-pool)',
    'default_logan_ewc_5_2kb': 'NT-50M (no overlap, logan, EWC 5, 2k ctx.)',
    'overlap_logan_ewc_5_pca_cls_256': 'NT-50M (overlap, logan, EWC 5, contrastive CLS)',
    'overlap_logan_ewc_5_pca_mean_256': 'NT-50M (overlap, logan, EWC 5, contrastive mean-pool)',
    'default_logan_ewc_5_pca_cls_256': 'NT-50M (no overlap, logan, EWC 5, contrastive CLS)',
    'default_logan_ewc_5_pca_mean_256': 'NT-50M (no overlap, logan, EWC 5, contrastive mean-pool)',
    'default_logan_ewc_5_sh_gc':  'NT-50M (no overlap, logan, EWC 5, GC & Shannon)',
    'default_logan_ewc_5_2kb_sh_gc':  'NT-50M (no overlap, logan, EWC 5, GC & Shannon, 2k ctx.)',
    'overlap_logan_ewc_5_sh_gc':  'NT-50M (overlap, logan, EWC 5, GC & Shannon)',
    'overlap_logan_ewc_5_2kb_sh_gc':  'NT-50M (overlap, logan, EWC 5, GC & Shannon, 2k ctx.)',
    'overlap_logan_ewc_5_2kb': 'NT-50M (overlap, logan, EWC 5, 2k ctx.)'
}

def get_for_all_compare_to_litereature(type: DATATYPE):
    group = [
            'default_multi_species_no_cont',
            'default_multi_species_no_cont_100',
            'default_multi_species_no_cont_250',
            'default_multi_species_no_cont_500',
            'default_multi_species',
            'default_multi_species_2kb',
            'overlap_multi_species',
            'overlap_multi_species_2kb',
            'overlap_logan_no_ewc',
            'overlap_logan_ewc_0_5',
            'overlap_logan_ewc_1',
            'overlap_logan_ewc_2',
            'overlap_logan_ewc_5',
            'overlap_logan_ewc_10',
            'overlap_logan_ewc_25',
            'default_multi_species_sh_gc',
            'default_multi_species_2kb_sh_gc',
            'overlap_multi_species_sh_gc',
            'overlap_multi_species_2kb_sh_gc',
            'default_multi_species_pca_cls_256',
            'default_multi_species_pca_mean_256',
            'overlap_multi_species_pca_cls_256',
            'overlap_multi_species_pca_mean_256',
            'default_logan_no_ewc',
            'default_logan_ewc_0_5',
            'default_logan_ewc_1',
            'default_logan_ewc_2',
            'default_logan_ewc_5',
            'default_logan_ewc_10',
            'default_logan_ewc_25',
            'default_logan_ewc_5_2kb',
            'overlap_logan_ewc_5_pca_cls_256',
            'overlap_logan_ewc_5_pca_mean_256',
            'default_logan_ewc_5_pca_cls_256',
            'default_logan_ewc_5_pca_mean_256',
            'default_logan_ewc_5_sh_gc',
            'default_logan_ewc_5_2kb_sh_gc',
            'overlap_logan_ewc_5_sh_gc',
            'overlap_logan_ewc_5_2kb_sh_gc',
            'overlap_logan_ewc_5_2kb'
    ]
    filename = 'compare_all_literature'

    if type == DATATYPE.TRAINING_CURVES:
        results = _collect_training_data(group)
        return results, filename
    elif type == DATATYPE.BENCHMARK:
        results = _collect_benchmark_data(group)
        return results, filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_all_compare(type: DATATYPE):
    group = [
            'default_multi_species',
            'default_multi_species_2kb',
            'overlap_multi_species',
            'overlap_multi_species_2kb',
            'overlap_logan_no_ewc',
            'overlap_logan_ewc_0_5',
            'overlap_logan_ewc_1',
            'overlap_logan_ewc_2',
            'overlap_logan_ewc_5',
            'overlap_logan_ewc_10',
            'overlap_logan_ewc_25',
            'default_multi_species_sh_gc',
            'default_multi_species_2kb_sh_gc',
            'overlap_multi_species_sh_gc',
            'overlap_multi_species_2kb_sh_gc',
            'overlap_multi_species_pca_cls_256',
            'overlap_multi_species_pca_mean_256',
            'default_logan_no_ewc',
            'default_logan_ewc_0_5',
            'default_logan_ewc_1',
            'default_logan_ewc_2',
            'default_logan_ewc_5',
            'default_logan_ewc_10',
            'default_logan_ewc_25',
            'overlap_logan_ewc_5_pca_cls_256',
            'overlap_logan_ewc_5_pca_mean_256',
            'default_logan_ewc_5_pca_cls_256',
            'default_logan_ewc_5_pca_mean_256',
            'default_logan_ewc_5_sh_gc',
            'default_logan_ewc_5_2kb_sh_gc',
            'overlap_logan_ewc_5_sh_gc',
            'overlap_logan_ewc_5_2kb_sh_gc',
            'overlap_logan_ewc_5_2kb'
    ]
    filename = 'compare_all'

    if type == DATATYPE.TRAINING_CURVES:
        results = _collect_training_data(group)
        return results, filename
    elif type == DATATYPE.BENCHMARK:
        results = _collect_benchmark_data(group)
        return results, filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_dataset_compare(type: DATATYPE):
    group = [
        'overlap_multi_species',
        'overlap_multi_species_2kb',
        'default_logan_ewc_5',
        'default_logan_ewc_5_2kb'
    ]
    filename = 'compare_context_length'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_context_length_compare(type: DATATYPE):
    group = [
        'overlap_multi_species',
        'overlap_multi_species_2kb',
        'default_logan_ewc_5',
        'default_logan_ewc_5_2kb'
    ]
    filename = 'compare_context_length'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_ewc_compare(type: DATATYPE):
    group = [
        'overlap_logan_no_ewc',
        'overlap_logan_ewc_0_5',
        'overlap_logan_ewc_1',
        'overlap_logan_ewc_2',
        'overlap_logan_ewc_5',
        'overlap_logan_ewc_10',
        'overlap_logan_ewc_25',
        'default_logan_no_ewc',
        'default_logan_ewc_0_5',
        'default_logan_ewc_1',
        'default_logan_ewc_2',
        'default_logan_ewc_5',
        'default_logan_ewc_10',
        'default_logan_ewc_25',
        'default_multi_species',
        'overlap_multi_species'
    ]
    filename = 'compare_ewc'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename



def get_for_best_logan_compare(type: DATATYPE):
    group = [
        'default_logan_ewc_5',
        'default_multi_species'
    ]
    filename = 'compare_logan'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_reference_compare(type: DATATYPE):
    group = [
        'ref_multi_species_no_cont_500_human',
        'ref_multi_species_no_cont_500_tg',
        'default_multi_species_no_cont_500_tg',
        'default_multi_species_no_cont_500_human'
    ]
    filename = 'compare_logan'

    if type == DATATYPE.TRAINING_CURVES:
        raise Exception("No training curves for reference comparison.")
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group, True), filename
    elif type == DATATYPE.UTR_CLASS:
        raise Exception("No UTR class results for reference comparison.")

def get_for_validation_compare(type: DATATYPE):
    group = [
        'ref_multi_species_no_cont_500_human',
        'ref_multi_species_no_cont_500_tg',
        'default_multi_species_no_cont_500_tg',
        'default_multi_species_no_cont_500_human',
        'default_multi_species_no_cont_random_init_500_tg',
        'default_multi_species_no_cont_random_init_500_human',
    ]
    filename = 'compare_logan'

    if type == DATATYPE.TRAINING_CURVES:
        raise Exception("No training curves for reference comparison.")
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group, True), filename
    elif type == DATATYPE.UTR_CLASS:
        raise Exception("No UTR class results for reference comparison.")

def get_for_pca_embedding_compare(type: DATATYPE):
    group = [
        'pca_cls_256_multi_species_overlap',
        'pca_mean_256_multi_species_overlap'
    ]
    filename = 'compare_pca_embedding'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_pca_dim_compare(type: DATATYPE):
    group = []
    filename = 'compare_pca_dimensions'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

def get_for_interesting_compare(type: DATATYPE):
    group = [
        'pca_cls_256_multi_species_overlap',
        'pca_mean_256_multi_species_overlap'
    ]
    filename = 'compare_pca_embedding'

    if type == DATATYPE.TRAINING_CURVES:
        return _collect_training_data(group), filename
    elif type == DATATYPE.BENCHMARK:
        return _collect_benchmark_data(group), filename
    elif type == DATATYPE.UTR_CLASS:
        return _collect_utr_class_data(group), filename

if __name__ == "__main__":
    check_model_existence()
    for m in MODELS:
        print(f"MODEL={m['name']} sh evaluate_trained.sh")