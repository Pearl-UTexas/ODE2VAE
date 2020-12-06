import logging
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Sequence

import tensorflow as tf  # type: ignore

from model.data.utils import plot_latent
from model.data.wrappers import *
from model.ode2vae_args import ODE2VAE_Args


def make_dirs(paths: Sequence[Path]):
    for path in paths:
        if not path.exists():
            path.mkdir()


def make_ext(args) -> str:
    if args.task == "bballs" or "mnist" in args.task:
        ext = f"{args.task:s}_q{args.q:d}_inst{args.inst_enc_KL:d}_fopt{args.f_opt:d}_enc{args.NF_enc:d}_dec{args.NF_dec:d}"
    elif "mocap" in args.task:
        ext = f"{args.task:s}_q{args.q:d}_inst{args.inst_enc_KL:d}_fopt{args.f_opt:d}_He{args.He:d}_Hf{args.Hf:d}_Hd{args.Hd:d}"
    else:
        raise ValueError(f"Invalid task {args.task}")
    logging.info("file extensions are {:s}".format(ext))
    return ext


def data_map(X, y, W, p=0, dt=0.1):
    W += tf.random_uniform([1], 0, 1, tf.int32)[0]  # needed for t to be of dim. None
    W = tf.cast(W, tf.int32)
    rng_ = tf.range(0, W)
    t_ = tf.to_float(dt) * tf.cast(rng_, tf.float32)
    X = tf.gather(X, rng_, axis=1)
    y = tf.gather(y, rng_, axis=1)
    return X, y, t_


def main():
    sess = tf.InteractiveSession()

    ########### setup params, data, etc ###########
    # read params
    args = ODE2VAE_Args().parse()
    logging.info(args)

    make_dirs([Path(args.ckpt_dir) / args.task, Path("plots") / args.task])

    # dataset
    dataset, N, T, D = load_data(
        args.data_root, args.task, subject_id=args.subject_id, plot=True
    )

    # artificial time points
    dt = 0.1
    t = dt * np.arange(0, T, dtype=np.float32)

    ext = make_ext(args)

    ########### training related stuff ###########
    xval_batch_size = int(args.batch_size / 2)
    min_val_lhood = -1e15

    xbspl = tf.placeholder(tf.int64, name="tr_batch_size")
    xfpl = tf.placeholder(tf.float32, [None, None, D], name="tr_features")
    xtpl = tf.placeholder(tf.float32, [None, None], name="tr_timepoints")

    partial_map = partial(data_map, W=T, dt=dt)

    xtr_dataset = (
        tf.data.Dataset.from_tensor_slices((xfpl, xtpl))
        .batch(xbspl)
        .map(data_map, 8)
        .prefetch(2)
    )
    xval_dataset = (
        tf.data.Dataset.from_tensor_slices((xfpl, xtpl))
        .batch(xbspl)
        .map(data_map, 8)
        .repeat()
    )

    xiter_ = tf.data.Iterator.from_structure(
        xtr_dataset.output_types, xtr_dataset.output_shapes
    )
    if "nonuniform" not in args.task:
        X, _, t = xiter_.get_next()
    else:
        X, t, _ = xiter_.get_next()
    xtr_init_op = xiter_.make_initializer(xtr_dataset)
    xval_init_op = xiter_.make_initializer(xval_dataset)

    ########### model ###########
    if "nonuniform" not in args.task:
        from model.ode2vae import ODE2VAE
    else:
        from model.ode2vae_nonuniform import ODE2VAE
    vae = ODE2VAE(
        sess,
        args.f_opt,
        args.q,
        D,
        X,
        t,
        NF_enc=args.NF_enc,
        NF_dec=args.NF_dec,
        KW_enc=args.KW_enc,
        KW_dec=args.KW_dec,
        Nf=args.Nf,
        Ne=args.Ne,
        Nd=args.Nd,
        task=args.task,
        eta=args.eta,
        L=1,
        Hf=args.Hf,
        He=args.He,
        Hd=args.Hd,
        activation_fn=args.activation_fn,
        inst_enc_KL=args.inst_enc_KL,
        amort_len=args.amort_len,
        gamma=args.gamma,
    )

    ########### training loop ###########
    t0 = time.time()

    print(
        "{:>15s}".format("epoch")
        + "{:>15s}".format("total_cost")
        + "{:>15s}".format("E[p(x|z)]")
        + "{:>15s}".format("E[p(z)]")
        + "{:>15s}".format("E[q(z)]")
        + "{:>16s}".format("E[KL[ode||enc]]")
        + "{:>15s}".format("valid_cost")
        + "{:>15s}".format("valid_error")
    )
    print(
        "{:>15s}".format("should")
        + "{:>15s}".format("decrease")
        + "{:>15s}".format("increase")
        + "{:>15s}".format("increase")
        + "{:>15s}".format("decrease")
        + "{:>16s}".format("decrease")
        + "{:>15s}".format("decrease")
        + "{:>15s}".format("decrease")
    )
    for epoch in range(args.num_epoch):
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        num_iter = 0
        Tss = max(min(T, T // 5 + epoch // 2), vae.amort_len + 1)
        sess.run(
            xtr_init_op,
            feed_dict={
                xfpl: dataset.train.x,
                xtpl: dataset.train.y,
                xbspl: args.batch_size,
            },
        )
        while True:
            try:
                if np.mod(num_iter, 20) == 0:
                    print(num_iter)
                ops_ = [
                    vae.vae_optimizer,
                    vae.vae_loss,
                    vae.reconstr_lhood,
                    vae.log_p,
                    vae.log_q,
                    vae.inst_enc_loss,
                ]
                values_batch = sess.run(ops_, feed_dict={vae.train: True, vae.Tss: Tss})
                values = [values[i] + values_batch[i + 1] for i in range(5)]
                num_iter += 1
            except tf.errors.OutOfRangeError:
                break
        values = [values[i] / num_iter for i in range(5)]
        xtr_dataset = xtr_dataset.shuffle(buffer_size=dataset.train.N)
        sess.run(
            xval_init_op,
            feed_dict={
                xfpl: dataset.val.x,
                xtpl: dataset.val.y,
                xbspl: xval_batch_size,
            },
        )
        val_lhood = 0
        num_val_iter = 10
        for _ in range(num_val_iter):
            try:
                val_lhood += sess.run(
                    vae.mean_reconstr_lhood, feed_dict={vae.train: False, vae.Tss: Tss}
                )
            except tf.errors.OutOfRangeError:
                break
        val_lhood = val_lhood / num_val_iter / Tss
        xval_dataset = xval_dataset.shuffle(buffer_size=dataset.val.N)

        if val_lhood > min_val_lhood:
            min_val_lhood = val_lhood
            vae.save_model(args.ckpt_dir, ext)
            X, ttr = dataset.train.next_batch(5)
            Xrec = vae.reconstruct(X, ttr)
            zt = vae.integrate(X, ttr)
            plot_reconstructions(
                args.task,
                X,
                Xrec,
                ttr,
                show=False,
                fname="plots/{:s}/rec_tr_{:s}.png".format(args.task, ext),
            )
            plot_latent(
                zt,
                vae.q,
                vae.L,
                show=False,
                fname="plots/{:s}/latent_tr_{:s}.png".format(args.task, ext),
            )
            X, tval = dataset.val.next_batch(5)
            Xrec = vae.reconstruct(X, tval)
            # zt   = vae.integrate(X)
            plot_reconstructions(
                args.task,
                X,
                Xrec,
                tval,
                show=False,
                fname="plots/{:s}/rec_val_{:s}.png".format(args.task, ext),
            )
            # plot_latent(zt,vae.q,vae.L,show=False,fname='plots/{:s}/latent_val_{:s}.png'.format(task,ext))
            val_err = -1
            if "mnist" in args.task:
                X1 = X[:, args.amort_len :, :]
                X2 = Xrec[:, args.amort_len :, :]
                val_err = np.mean((X1 - X2) ** 2)
            elif args.task == "bballs":
                X1 = X[:, args.amort_len : args.amort_len + 10, :]
                X2 = Xrec[:, args.amort_len : args.amort_len + 10, :]
                val_err = np.sum((X1 - X2) ** 2, 2)
                val_err = np.mean(val_err)
            elif args.task == "mocap_single":
                diff = X[0, :, :] - Xrec[0, :, :]
                diff = diff[4 * diff.shape[0] // 5 :, :] ** 2
                val_err = np.mean(diff)
            elif args.task == "mocap_many":
                val_err = np.mean((X - Xrec) ** 2)
            print(
                "{:>15d}".format(epoch)
                + "{:>15.1f}".format(values[0])
                + "{:>15.1f}".format(values[1])
                + "{:>15.1f}".format(values[2])
                + "{:>15.1f}".format(-values[3])
                + "{:>15.1f}".format(values[4])
                + "{:>15.1f}".format(val_lhood)
                + "{:>15.3f}".format(val_err)
            )
        else:
            print(
                "{:>15d}".format(epoch)
                + "{:>15.1f}".format(values[0])
                + "{:>15.1f}".format(values[1])
                + "{:>15.1f}".format(values[2])
                + "{:>15.1f}".format(-values[3])
                + "{:>15.1f}".format(values[4])
                + "{:>15.1f}".format(val_lhood)
            )

        if math.isnan(values[0]):
            print("*** average cost is nan. terminating...")
            break

    t1 = time.time()
    print("elapsed time: {:.2f}".format(t1 - t0))


if __name__ == "__main__":
    main()
