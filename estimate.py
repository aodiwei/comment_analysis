#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/7/20'
# 
"""
import os
import csv

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import define
from tools.utility import Utility
from works.work_botnet_supervise.domain_cnn import data_process

log = Utility.get_logger('ml_cnn_es_domain')
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = Utility.conf_get('botnet')
model_folder = os.path.join(config.get('root_path'), config.get('model_folder'), config.get('cur_cnn_model_name'))
checkpoint_dir = os.path.join(model_folder, 'checkpoints')

# Parameters
# ==================================================
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_boolean("valid", False, "Evaluate on all valid data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
log.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    log.info("{}={}".format(attr.upper(), value))
log.info("")


def estimate():
    """
    评估，也可以再训练
    :return: 
    """
    # Load data
    if FLAGS.eval_train:
        x_raw, y_test, _ = data_process.load_data_and_labels()
        y_test = np.argmax(y_test, axis=1)
    elif FLAGS.valid:
        x_test, y_test = data_process.load_valid_data()
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw_o = [
            'csdnimg.cn',
            'msftncsi.com',
            '3vasqlg5ewraibsi.onion',
            'msftconnecttest.com',
            '2e0a24317f4a9294563f-26c3b154822345d9dde0204930c49e9c.ssl.cf1.rackcdn.com',
            '1grauqmt1ghzdrpbtfa3dommrgrzgramjct4zya5uf3ts65e.ourdvsss.com',
            '4gr5ukmtwgczdrcbqgc5o.ourdvsss.com',
            'aw.kejet.net',
            'bookq.cn',
            'chinawbsyxh.com',
            'coomtawmaeooeeototaobnmawerefehwewretrerwrnuooecdo.ggytc.com',
            'cqyywh.com',
            'gfsousou.cn',
            'gzchq.cn',
            'hanqidq.com',
            'icnfnt.com',
            'kmhlkq.com',
            'lyskqyy.com',
            'm.kejet.net',
            'microfinancefund.cn',
            'qunazhu.cn',
            'renaikouqiang.com',
            'srtlc.cn',
            'sync.rhythmxchange.com',
            'xmbxmy.com',
            'xmdtkq.com',
            'ybskq.cn',
            'update.sdk.jiguang.cn',
            'view.bug.cn',
            'vip.zfxfsz.cn',
            'tracking.prf.hn',
            'ttfo.ykylt.cn',
            'tool.lu',
            'tool.php.cn',
            'topic.qixumu.cn',
            'szse.cn',
            'thumb.jfcdns.com',
            'tj.ejie.me',
            'tmoblbssdk.yy.duowan.com',
            'tool.lu',
            'sugs.m.sm.cn',
            's.cn',
            '25z5g623wpqpdwis.onion.top',
            'l28pxmxiyisl28prgtcrhrhuc49cvo21cud10.net',
            'sajajlyoogrmkjlkmosbxowcrmwlvajdkbtbjoy.nylalobghyhirgh.com',
            'sajajlyoogrmkkmhncrjkingvmwlvajdketeknvbwfqppgkbtdlcj.esjsnwhjmjglnoksjmctgrlyhsgmgveqmrexmloppylmpl.nylalobghyhirgh.com',
            'sajajlyoogrmkpmnmixivemirmwlvajdkctcjpymyjlfmoqjyaqplm.tfvduaplkilcogrcpbv.nylalobghyhirgh.com',
            'sajajlyoogrmkdjhrgpcllwanowlvajdkftfjcxlyokpmancxmqnpkrnwdx.dlpqjnholroqctarosbtpq.nylalobghyhirgh.com',
            'sajajlyoogrmkjjmjmmhjdkgmmwlvajdkjtcmiycxjlppolisfqgpcs.jsnwap.nylalobghyhirgh.com',
            'sajajlyoogrmkpmnmixivemirmwlvajdkctcjpymyjlfmoqjyaqplmtfvduap.lkilcogrcpbv.nylalobghyhirgh.com',
            'sajajlyoogrmkpmnmixivemirmwlvajdkctcjpymyjlfmoqjyaqplmtfv.duaplkilcogrcpbv.nylalobghyhirgh.com',
            'sajajlyoogrmkeloufodqfpjwmwlvajdkctmkcydybloooljwaqpp.gsoskwdkljlmkoksiqduix.nylalobghyhirgh.com',
            'sajajlyoogrmkdmkporgujqmumwlvajdkctgjewiufqoppkotelgmovfvexem.lmaklmoxgoftfrcsbtgkayiohuevhknnevkj.nylalobghyhirgh.com',
            'sajajlyoogrmkmliwgmgoooavmwlvajdkctckcwgvmjkjbpivjmgmc.udvnyamjmmjlmoxhvaphjencqasmmbsfv.nylalobghyhirgh.com',
            'sajajlyoogrmkglhsnqnmkkpqmwlvajdkctckcwgvmjkjbpivjmgmcudvnyamj.mmjlmoxhvaphjencqasmmbsfv.nylalobghyhirgh.com',
            'sajajlyoogrmkekbsbnowiwnsmwlvajdkctomcymyklhmdjpxbplqkrb.snwekokgllmoxapeubsorotbkhynnktft.nylalobghyhirgh.com',
            'sajajlyoogrmkdjhrgpcllwanowlvajdkftfjcxlyokpmancxmqnpkrnwdxdl.pqjnholroqctarosbtpq.nylalobghyhirgh.com',
            'sajajlyoogrmklkjqgxdxbxiymwlvajdkctckcwgvmjkjbpivjmgmc.udvnyamjmmjlmoxhvaphjencqasmmbsfv.nylalobghyhirgh.com',
            'sajajlyoogrmkpmnmixivemirmwlvajdkctcjpymyjlfmoqjyaq.plmtfvduaplkilcogrcpbv.nylalobghyhirgh.com']
        # x_raw = [' '.join(list(x)) for x in x_raw_o]

        x_raw_o = [
            'csdnimg.cn',
            'ifc.wps.cn',
            '1ic.wps.cn',
            '1grhdymt3ghzdnqbofa3dommrgrzgramjct4zya5uf3ts65e.ourdvsss.com',
            '1grhdymt3ghzdnqbofa3u1mmrgrzgramjct4zya5uf3ts65e.ourdvsss.com',
            '2059275571.cdnle.net',
            '27lelchgcvs2wpm7.xmvr54.to',
            '2e0a24317f4a9294563f-26c3b154822345d9dde0204930c49e9c.ssl.cf1.rackcdn.com',
            '3g.dlbyqc.cn',
            '3gc5nhctigyzdqptqgr5ug.ourdvsss.com',
            '3gc5nhctigyzdqptqgrhdy.ourdvsss.com',
            '4bfeccfaedfeaa5af677-61b6e47827637cef7adda99fc8c55c47.ssl.cf1.rackcdn.com',
            '4gaa1hcjsgyzdrcb3fa4dgmmdctzdycbqcjos13diqbtzgmudp7so.ourdvsss.com',
            '4gaa1hpjwfa3dnqjqge3y.ourdvsss.com',
            '4gc5nhctw8yzdncbqgaay.ourdvsss.com',
            '4gr5ukmtwgczdrcbqgc5o.ourdvsss.com',
            '554vb025qhze1381.gfvip07as.com',
            '7f9c61237bd6e732e57e-5fa18836a2ae6b5e7c49abcc89b20237.ssl.cf1.rackcdn.com',
            '8c7ae97a12937bfd99be-84c167a1847370a4ed20bcbb479190e8.ssl.cf1.rackcdn.com',
            '9.rarbg.to',
            'a13ac0000579edbd1f5b5a5cec2eb3ecd.profile.atl52.cloudfront.net',
            'a4.wzzldn.cn',
            'aa4fcf95f3e4fb2434d242869ee1720cf.profile.lhr3.cloudfront.net',
            'aa8fe1459d6dbb5fef95672fa1c4e8133.profile.cdg50.cloudfront.net',
            'adashx4yt.m.taobao.com',
            'afb75eeb654b61511b8dacf99e15fa39f.profile.gig50.cloudfront.net',
            'aimr.to',
            'anhuiqq.cn',
            'apk.szkdwh.cn',
            'awxy.cn',
            'babeljs.cn',
            'beej.us',
            'beijingtuku.cn',
            'bj.imp.voiceads.cn',
            'bjjubao.org',
            'bookq.cn',
            'bootcdn.cn',
            'ccjqdjlgwvfjdr.wh.sapling.com',
            'cdn.infoqstatic.com',
            'cdn.wiz.cn',
            'cdns.eu1.gigya.com',
            'chaoyang166.cn',
            'chinajnhb.com',
            'chinawbsyxh.com',
            'cjrb.cjn.cn',
            'cm.dmp.sina.cn',
            'cnfffff.com',
            'cnzhx.net',
            'coomtawmaeooeeototaobnmawerefehwewretrerwrnuooecdo.ggytc.com',
            'copyto.cn',
            'corrupteddevelopment.com',
            'cqgseb.cn',
            'cqtimes.cn',
            'cqyywh.com',
            'creatim.qtmojo.cn',
            'ctcdns.cn',
            'cvlab.epfl.ch',
            'cyidc.cc',
            'd1cedvxeo5ngwa.cloudfront.net',
            'd1ophd2rlqbanb.cloudfront.net',
            'd1xfq2052q7thw.cloudfront.net',
            'd2wsq0nhucb4jg.cloudfront.net',
            'd3ui957tjb5bqd.cloudfront.net',
            'dahuaabfw.com',
            'diffeao.cn',
            'difnxm.cn',
            'diyvm.com',
            'dizhi.xin',
            'dl.meizu.cn',
            'douohjwkqkcnnys.wh.sapling.com',
            'down.cn',
            'down.zvvdh.to',
            'dtlilztwypawv.cloudfront.net',
            'dxtsgxb.cn',
            'eb.yixin.im',
            'eb3456ad9d8b827bea09fa81cb6de93d.7cname.com',
            'ebzj.reg.163.com',
            'epwqlsxnawqndy.iteye.com',
            'faxingw.cn',
            'fengj.cn',
            'fff.ly',
            'fg8vvsvnieiv3ej16jby.litix.io',
            'fuwuqun.win',
            'fwqtg.net',
            'gfsousou.cn',
            'glutnn.cn',
            'gwwdc.adtddns.asia',
            'gzchq.cn',
            'haorj.cn',
            'hbhxqq.com',
            'hbjhgczxyxgsycfgs.21hubei.com',
            'hbsrsksy.cn',
            'hcb.cjn.cn',
            'hgjj.cn',
            'hlszws.com',
            'hmlnx.cn',
            'hszqzj.com',
            'hwb.cjn.cn',
            'iapps.im',
            'icnfnt.com',
            'img.php.cn',
            'inxt.hzbdu.cn',
            'jlck.cn',
            'kbhbeijingzongbu.com',
            'kbhbjgwzx.com',
            'kbtq.cn',
            'khtmltopdf.org',
            'kjjl100.cn',
            'kmhlkq.com',
            'kuwo.cn',
            'laihj.net',
            'linuxdiyf.com',
            'lkme.cc',
            'lstsrw.org',
            'lyskqyy.com',
            'lyztdz.com',
            'marry.xizi.com',
            'mcdn.qtmojo.cn',
            'microfinancefund.cn',
            'mvnrepository.com',
            'myfish.cc',
            'neuralnetworksanddeeplearning.com',
            'o0s106hgi.qnssl.com',
            'o3e85j0cv.qnssl.com',
            'oicqzone.com',
            'onbdsxoacyqbpek.wh.sapling.com',
            'only-642391-58-19-41-33.nstool.net',
            'pay.swiftpass.cn',
            'pdfwork.cn',
            'pgsqldb.org',
            'phpxuexi.com',
            'piwik.ulo.pe',
            'pv0kcit457157z6.aligaofang.com',
            'q.infoqstatic.com',
            'qa-www.snow.me',
            'qdcqlv.com',
            'qdrkcdqgcaz.china.herostart.com',
            'qdyyjiazheng.com',
            'qdztdqzdh.china.herostart.com',
            'qdztydqgc.china.herostart.com',
            'qhdxw.com',
            'qikbo.cn',
            'qixumu.cn',
            'qmlog.cn',
            'qnr.cn',
            'qqbaobao.com',
            'qqhuo.cn',
            'qr.wps.cn',
            'qstheory.cn',
            'qunazhu.cn',
            'qunxionglm.com',
            'renaikouqiang.com',
            'runjs.cn',
            'sdhuiyin.cn',
            'serv.vip.dns.iqiyi.com',
            'shjbzx.cn',
            'slfyflgc.cn',
            'softjie.cn',
            'sqlcipher.net',
            'srtlc.cn',
            'sshcdhpjfh.jin10.com',
            'sshcdhpjnm.jin10.com',
            'static.vux.li',
            'sync.rhythmxchange.com',
            'tgcep.cn',
            'thinkphp.cn',
            'thumb.jfcdns.com',
            'tmoblbssdk.yy.duowan.com',
            'tool.lu',
            'tool.php.cn',
            'topic.qixumu.cn',
            'tortoisesvn.net',
            'u996.v.bsgslb.cn',
            'uspat.cc',
            'uyan.cc',
            'view.bug.cn',
            'vip.zfxfsz.cn',
            'vuejs.org',
            'weibo.cn',
            'wkhtmltopdf.org',
            'xcf.cn',
            'xielw.cn',
            'xmbxmy.com',
            'xmdtkq.com',
            'xnimg.cn',
            'xueui.cn',
            'xxkhh.com',
            'ybskq.cn',
            'ycbxtxwlyxgs.21hubei.com',
            'yccsmtkjkfyxgs.21hubei.com',
            'ycdfjzlwyxgs.21hubei.com',
            'ycfsjsmyxgs.21hubei.com',
            'ychlwyglyxgs928.21hubei.com',
            'ycjhftzglyxgs.21hubei.com',
            'ycsbttxkjyxgs.21hubei.com',
            'ycslxhbjnkjyxgs.21hubei.com',
            'ycsndlyxwhyxgs.21hubei.com',
            'ycsqhyhhyxgs.21hubei.com',
            'ycsxylwyxgs118.21hubei.com',
            'ycyffsqcfwyxgs.21hubei.com',
            'yczywlkjyxgs.21hubei.com',
            'yjrddh.cn',
            'yjsglc.bjwlxy.cn',
            'yzfktdq.huisou.com',
            'yzjldq.com',
            'zgjjzk.cn',
            'zgswcn.com',
            'zhaizhouwei.cn',
            'zhuzi.me',
            'zkx.cc',
        ]
        Utility.init_alphabet_dict()
        x_test = [Utility.pre_encode_domain(x)[1] for x in x_raw_o]
        y_test = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        y_test = [0] * 210

    # Map data into vocabulary
    # vocab_file = config.get('vocab')
    # vocab_path = os.path.join(define.root, 'src', 'static_data', vocab_file)
    # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # x_test = np.array(list(vocab_processor.transform(x_raw)))

    log.info("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    # checkpoint_dir = r'E:\home\adw\data\dis_model\checkpoints'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_process.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # log.info accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        log.info("Total number of test examples: {}".format(len(y_test)))
        log.info("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_raw_o), y_test, all_predictions))
        for x in predictions_human_readable:
            if int(float(x[2])) == 1:
                print('{} is malicious domain'.format(x[0]))
    else:
        predictions_human_readable = np.column_stack((np.array(x_raw_o), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    log.info("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w', newline='') as f:
        csv.writer(f).writerows(predictions_human_readable)


if __name__ == '__main__':
    estimate()
