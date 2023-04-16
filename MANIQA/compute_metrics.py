import pandas as pd
import os 
import scipy
import logging
import sys
from config import Config

def main(config, net_filename='', output_filename="output_model_maniqa__seed83532.txt"):
    
    # output to file
    # logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO, 
    )
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
       
    fileHandler = logging.FileHandler("{0}/{1}.log".format(config.valid_path, f"report__{config.db_name}_{net_filename}"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    logging.info('Start')
   
    
    mos = pd.read_csv(config.test_mos_path, names=['mos'], sep=',')['mos'].values
    preds = pd.read_csv(os.path.join(config.valid_path, output_filename), names=['filename', 'predict'], sep=',')['predict'].values
    
    logging.info(mos[:6])
    logging.info(preds[:6])
    
    # SRCC
    SRCC = scipy.stats.spearmanr(a=mos, b=preds)
    logging.info(SRCC)
    logging.info('SRCC of MANIQA on {} = {:.2f}'.format(config.db_name, abs(SRCC.correlation)))
    # PLCC
    PLCC = scipy.stats.pearsonr(x=mos, y=preds)
    logging.info(PLCC)
    logging.info('PLCC of MANIQA on {} = {:.2f}'.format(config.db_name, abs(PLCC.correlation)))
    
if __name__ == "__main__":
    config = Config({
            
            "db_name": "TID2013",
            
            "test_mos_path": r"C:\Users\MQTyor\ai_pc\Reserch_ai\IQA\datasets\tid2013\mos.txt",                      
            "valid_path": "output/valid/inf_tid_table4_5seeds_pipal_fivePointCrop5",
        })
        
    main(config)