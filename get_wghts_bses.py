import tensorflow as tf
import scipy as sp
from scipy import io
import sys

def read_global_vars(chckpt_path):
    saver = tf.train.import_meta_graph(chckpt_path)
    sess = tf.Session()
    saver.restore(sess,tf.train.latest_checkpoint('/'.join(chckpt_path.split('/')[:-1])))
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    

    var_dict={v.name.replace('/','_').replace(':','_'):sess.run(v.name) for v in vars}

    return var_dict


def write_to_mat(ipdict,fname):
	io.savemat(fname,ipdict,do_compression=True)

if __name__ == '__main__':
	
	chckpt_path = sys.argv[1]
	opname = sys.argv[2]

	arrs = read_global_vars(chckpt_path)

	write_to_mat(arrs,opname)

	print('Objects from:',chckpt_path,'written to:',opname)
