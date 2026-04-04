def debug_loss(key, params, wires, x, y):
    print()

    # TODO figure out why he gets the dataset again
    # x_test, y_test = get_conway()
    # x_test, y_test = get_ttt()

    # train_loss = loss(params, wires, x, y, False)
    # test_loss = loss(params, wires, x_test, y_test, False)
    # test_loss_hard = loss(params, wires, x_test, y_test, True)

    train_acc = acc(params, wires, x, y)

    # preds = predict_batch(params, wires, x_test, False)
    # preds_hard = predict_batch(params, wires, x_test, True)
    # print("[", *[f"{x:.3g}" for x in preds[0:5].flatten().tolist()], "]", preds_hard[0:5].flatten(), y_test[0:5].flatten())
    print(f"train_acc: {train_acc*100:.3f} %")
    # print(f"train_loss: {train_loss:.3g}", end="; ")
    # print(f"test_loss: {test_loss:.3g}", end="; ")
    # print(f"test_loss_hard: {test_loss_hard:.3g}")