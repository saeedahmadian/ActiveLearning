import matplotlib.pyplot as plt
from UncertaintyPooling import UncertaintyPoolAL
import argparse

parser = argparse.ArgumentParser('Active Learning with Uncertainty Pool-based sampling')
parser.add_argument("--epochs", default=2, help="Number of epochs",
                    type=int)
parser.add_argument("--batch_size", default=32, help="batch size",
                    type=int)
parser.add_argument("--lr", default=32, help="learning rate",
                    type=float)
parser.add_argument("--sampling_limit", default=5, help="sampling limit",
                    type=int)
parser.add_argument("--pooling_ratio", default=.4, help="ratio of to be labeled",
                    type=float)
parser.add_argument("--total_ratio", default=5, help="ratio of total data size",
                    type=float)

args = parser.parse_args()

if __name__ == '__main__':
    mysampling= UncertaintyPoolAL(total_ratio=args.total_ratio,pooling_ratio=args.pooling_ratio,
                                  batch_size=args.batch_size,sampling_limit=args.sampling_limit,lr=args.lr)
    bench_mark_accuracy = mysampling.bench_mark()
    ratio_confidence_accuracy = mysampling.Run_experiment('ratio-confidence')
    entropy_accuracy = mysampling.Run_experiment('entropy')
    least_confidence_accuracy = mysampling.Run_experiment('least-confidence')
    fig,ax= plt.subplots(1,1,figsize=(12,8))
    ax.plot([bench_mark_accuracy for _ in range(args.sampling_limit)],label='trained over entire dataset')
    ax.plot(ratio_confidence_accuracy,label='ratio-confidence')
    ax.plot(entropy_accuracy, label='entropy')
    ax.plot(least_confidence_accuracy, label='least-confidence')
    ax.set_title('{} epochs, {} batch size, {} learning rate, to be labeled data % {}'.
                  format(args.epochs,args.batch_size,args.lr,100*args.pooling_ratio))
    ax.set_xlabel('iteration')
    ax.set_ylabel('Accuracy %')
    ax.legend()
    fig.savefig('final_results_uncertainty_pooling.png')
    fig.show()




