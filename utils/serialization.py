torch.save(
    {
        'epoch': epochID + 1,
        'state_dict': model.state_dict(),
        'best_loss': lossMIN,
        'optimizer': optimizer.state_dict(),
        'alpha': loss.alpha,
        'gamma': loss.gamma
    }, checkpoint_path + '/m-' + launchTimestamp + '-' +
    str("%.4f" % lossMIN) + '.pth.tar')
