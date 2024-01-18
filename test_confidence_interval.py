import time
from logging import getLogger
import pandas as pd
from daisy.model.MFRecommender import MF
from daisy.model.FMRecommender import FM
from daisy.model.NFMRecommender import NFM
from daisy.model.NGCFRecommender import NGCF
from daisy.model.EASERecommender import EASE
from daisy.model.SLiMRecommender import SLiM
from daisy.model.VAECFRecommender import VAECF
from daisy.model.NeuMFRecommender import NeuMF
from daisy.model.PopRecommender import MostPop
from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.model.PureSVDRecommender import PureSVD
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.splitter import TestSplitter
from daisy.utils.metrics import calc_ranking_results
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix
import numpy as np

model_config = {
    'mostpop': MostPop,
    'slim': SLiM,
    'itemknn': ItemKNNCF,
    'puresvd': PureSVD,
    'mf': MF,
    'fm': FM,
    'ngcf': NGCF,
    'neumf': NeuMF,
    'nfm': NFM,
    'multi-vae': VAECF,
    'item2vec': Item2Vec,
    'ease': EASE,
    'lightgcn': LightGCN,
}

if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config()

    ''' init seed for reproducibility '''
    results_5 = []
    results_10 = []
    results_20 = []
    for seed in range(2022, 2022 + 3):
        config['seed'] = seed
        init_seed(config['seed'], False)
        #init_seed(config['seed'], config['reproducibility'])

        ''' init logger '''
        init_logger(config)
        logger = getLogger()
        logger.info(config)
        config['logger'] = logger

        ''' Test Process for Metrics Exporting '''
        reader, processor = RawDataReader(config), Preprocessor(config)
        df = reader.get_data()
        df = processor.process(df)
        user_num, item_num = processor.user_num, processor.item_num

        config['user_num'] = user_num
        config['item_num'] = item_num

        ''' Train Test split '''
        splitter = TestSplitter(config)
        train_index, test_index = splitter.split(df)
        train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

        # get rated data
        raw_rated = dict(train_set.groupby(by=config['UID_NAME'])[config['IID_NAME']].unique())
        rated = {}
        for user in raw_rated:
            rated[user] = set(raw_rated[user])

        # get parameters
        best_params = pd.read_csv(
            'tune_res/best_params_BPR_' + config['algo_name'] + '_' + config['dataset'] + '_origin_tloo_' + config['path'])
        for col in best_params.columns[:-1]:
            config[col] = best_params.loc[0, col]

        ''' get ground truth '''
        test_ur = get_ur(test_set)
        total_train_ur = get_ur(train_set)
        config['train_ur'] = total_train_ur


        ''' calculating KPIs '''
        logger.info('Save metric@k result to res folder...')
        result_save_path = f"./res/{config['dataset']}/{config['prepro']}/{config['test_method']}/confidence_interval/"
        algo_prefix = f"{config['loss_type']}_{config['algo_name']}"
        common_prefix = f"with_{config['sample_ratio']}{config['sample_method']}"

        ensure_dir(result_save_path)
        config['res_path'] = result_save_path


        ''' build and train model '''
        s_time = time.time()
        if config['algo_name'].lower() in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
            model = model_config[config['algo_name']](config)
            model.fit(train_set)

        elif config['algo_name'].lower() in ['multi-vae']:
            history_item_id, history_item_value, _ = get_history_matrix(train_set, config, row='user')
            config['history_item_id'], config['history_item_value'] = history_item_id, history_item_value
            model = model_config[config['algo_name']](config)
            train_dataset = AEDataset(train_set, yield_col=config['UID_NAME'])
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)

        elif config['algo_name'].lower() in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
            if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
                config['inter_matrix'] = get_inter_matrix(train_set, config)
            model = model_config[config['algo_name']](config)
            sampler = BasicNegtiveSampler(train_set, config)
            train_samples = sampler.sampling()
            train_dataset = BasicDataset(train_samples)
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)

        elif config['algo_name'].lower() in ['item2vec']:
            model = model_config[config['algo_name']](config)
            sampler = SkipGramNegativeSampler(train_set, config)
            train_samples = sampler.sampling()
            train_dataset = BasicDataset(train_samples)
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)

        else:
            raise NotImplementedError('Something went wrong when building and training...')
        elapsed_time = time.time() - s_time
        logger.info(
            f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

        ''' build candidates set '''
        logger.info('Start Calculating Metrics...')
        test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)

        ''' get predict result '''
        logger.info('==========================')
        logger.info('Generate recommend list...')
        logger.info('==========================')
        test_dataset = CandidatesDataset(test_ucands)
        test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        print('number of test: ', len(test_u))
        preds = model.full_rank(test_u)  # np.array (u, topk)
        # preds: [num_users, num_items]
        final_preds = []  # preds after filtering
        config['topk'] = 20
        for user_index in range(len(test_u)):
            user = test_u[user_index]
            recommended = []
            index = 0
            while len(recommended) != config['topk']:
                if preds[user_index, index] not in rated[user]:
                    recommended.append(preds[user_index, index])
                index += 1
            final_preds.append(np.asarray(recommended))
        final_preds = np.asarray(final_preds)
        results = calc_ranking_results(test_ur, final_preds, test_u, config)
        results_5.append([seed]+list(results[5].values))
        results_10.append([seed] + list(results[10].values))
        results_20.append([seed] + list(results[20].values))





    path = config['path']
    results_5_df = pd.DataFrame(results_5,columns = ['seed','Recall','MRR', 'NDCG','HR','Precision'])
    results_10_df = pd.DataFrame(results_10,columns = ['seed','Recall','MRR', 'NDCG','HR','Precision'])
    results_20_df = pd.DataFrame(results_20,columns = ['seed','Recall','MRR', 'NDCG','HR','Precision'])

    results_5_df.to_csv(f'{result_save_path}{algo_prefix}_{common_prefix}_top5_{path}', index=False)
    results_10_df.to_csv(f'{result_save_path}{algo_prefix}_{common_prefix}_top10_{path}', index=False)
    results_20_df.to_csv(f'{result_save_path}{algo_prefix}_{common_prefix}_top20_{path}', index=False)
