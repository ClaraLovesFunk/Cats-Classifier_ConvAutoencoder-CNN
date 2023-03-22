# ganzer autoencoder
model

# autoencoder
model.encoder(daten)

# autoen + clfhead
model_clf_head(model.encoder(daten))









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
