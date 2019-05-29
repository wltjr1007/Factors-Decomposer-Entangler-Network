import traceback
import os
import data_load
import settings as st
from zipfile import ZipFile
from glob import glob
import shutil
from datetime import datetime

if __name__ == "__main__":
    summ_path = st.result_path + "%d/%s/cur_%s/" % (st.mode, st.dataset_dict[st.dataset], datetime.now().strftime("%m%d_%H%M%S%f")[:-3])
    if st.part_mode==1:
        summ_path = summ_path[:-1]+"_%d_%d/"%(st.c_way, st.k_shot)

    if not os.path.exists(summ_path):
        os.makedirs(summ_path)

    zip_fn = ZipFile(summ_path + "code.zip", "w")
    for fn in glob(st.project_path + "**/*.py", recursive=True):
        zip_fn.write(filename=fn)
    log = ", ".join("%s: %s" % item for item in vars(st).items() if not item[0].startswith("__"))
    zip_fn.writestr("logs.txt", log)
    zip_fn.close()

    trn_dat, trn_lbl, one_dat, tst_dat, tst_lbl = data_load.load_data()
    try:
        import train_op
        import test_op

        Trainer = train_op.Trainer(trn_dat=trn_dat, trn_lbl=trn_lbl, one_dat=one_dat, tst_dat=tst_dat, tst_lbl=tst_lbl, summ_path=summ_path)
        Trainer.train()

        if os.path.exists(Trainer.summ_path) and "cur_" in Trainer.summ_path:
            shutil.move(Trainer.summ_path, Trainer.summ_path.replace("cur_", "fin_"))

    except KeyboardInterrupt:
        try:
            Trainer.train_summary_writer.close()
        except:
            pass
        if os.path.exists(summ_path) and "cur_" in summ_path:
            shutil.move(summ_path, summ_path.replace("cur_", "key_"))
        print(traceback.format_exc())
    except:
        try:
            Trainer.train_summary_writer.close()
        except:
            pass
        if os.path.exists(summ_path) and "cur_" in summ_path:
            shutil.move(summ_path, summ_path.replace("cur_", "err_"))
        print(traceback.format_exc())
