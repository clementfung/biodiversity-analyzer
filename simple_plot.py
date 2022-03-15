import matplotlib.pyplot as plt
import numpy as np

loss_obj = np.load('history.npy')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(np.arange(10) + 1, loss_obj, lw=2)

ax.set_ylim([0, 5])
ax.set_xlim([0, 11])

ax.set_title('Early Training Results', fontsize=20)
ax.set_xlabel('Training Epochs', fontsize=16)
ax.set_ylabel('Training Loss', fontsize=16)

fig.tight_layout()
plt.savefig('trainloss.pdf')