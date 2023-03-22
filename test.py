# works for black and white -- suspiciously good autoencoder for hardly any training

for (img, _) in train_loader: # iterating over the batches in train_loader

    recon = model(img)
    print(recon)

    images_recon_new = recon #output[0][2]
    print(f'size output: {images_recon_new.size()}')

    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = img.detach().numpy()
    recon = images_recon_new.detach().numpy()

    for i, item in enumerate(img):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])


    '''

dataiter = iter(train_loader) # same but for one batch?
img, labels = dataiter.next()

#recon = model(img)

imgs = img.detach().numpy()

#recon = recon.detach().numpy()

# plot original imgs
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    imshow(imgs[idx])
    print(labels[idx])
    ax.set_title(classes[labels[idx]])

plt.show() '''
