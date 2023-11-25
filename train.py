import argparse
import os

import numpy as np
import torch.utils.data as data

from copy import deepcopy
from datasets import ImgFile
from model.label_module import LabelModule
from model.autoencoder import AutoEncoderGcnModule
from model.fusion_module import *
from evaluate import *
from losses import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mscoco')
    parser.add_argument('--batch_size', type=int, default=500, help='number of images in a batch')
    parser.add_argument('--bit', type=int, default=128, help='length of hash codes')
    parser.add_argument('--Epoch_num', type=int, default=10, help='num of Epochs')
    parser.add_argument('--times', type=int, default=1, help='num of times')
    parser.add_argument('--nc', type=int, default=6000, help='complete pairs')
    parser.add_argument('--n1u', type=int, default=2000, help='incomplete pairs with only images')
    parser.add_argument('--n2u', type=int, default=2000, help='incomplete pairs with only texts')
    parser.add_argument('--gamma', type=float, default=0.5, help='balance the importance of image/text')
    parser.add_argument('--lamda', type=float, default=50, help='lamda')
    parser.add_argument('--a', type=float, default=0.6, help='a')
    parser.add_argument('--alpha', type=int, default=14, help='alpha')
    parser.add_argument('--beta', type=float, default=0.0000001, help='beta')
    parser.add_argument('--p1', type=float, default=0.4, help='node itself')
    parser.add_argument('--p2', type=float, default=0.3, help='node itself')
    parser.add_argument('--c', type=float, default=1, help='GCN propogation')

    return parser.parse_args()


def generate_train_ds(images, texts, labels):
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    index1 = index[0:args.nc]
    index2 = index[args.nc:(args.nc + args.n1u)]
    index3 = index[(args.nc + args.n1u):(args.nc + args.n1u + args.n2u)]
    image1 = images[index1]
    image1 = normalize(image1)
    text1 = texts[index1]
    label1 = labels[index1]
    image2 = images[index2]
    image2 = normalize(image2)
    label2 = labels[index2]
    text3 = texts[index3]
    label3 = labels[index3]

    # The mean values of existing image and text features are used to fill in the missing parts
    mean_text = np.mean(np.concatenate([text1, text3], axis=0), axis=0)
    mean_text = mean_text.reshape(1, len(mean_text))
    text2 = np.tile(mean_text, (args.n1u, 1))
    mean_image = np.mean(np.concatenate([image1, image2], axis=0), axis=0)
    mean_image = mean_image.reshape(1, len(mean_image))
    image1 = image1 - mean_image
    image2 = image2 - mean_image
    image3 = np.zeros((args.n2u, images.shape[1])).astype(np.float32)

    # All the features after completion
    images = np.concatenate([image1, image2, image3], axis=0)
    texts = np.concatenate([text1, text2, text3], axis=0)
    labels = np.concatenate([label1, label2, label3], axis=0)
    index = np.arange(labels.shape[0])
    np.random.shuffle(index)
    images = images[index]
    texts = texts[index]
    labels = labels[index]

    M1 = np.expand_dims((index < (args.nc + args.n1u)).astype(np.float32), axis=1)
    M2 = np.expand_dims((index < args.nc).astype(np.float32) + (index >= (args.nc + args.n1u)).astype(np.float32), axis=1)

    datasets = ImgFile(images, texts, labels, M1, M2)
    data_loader = data.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=True, num_workers=5)
    return data_loader, torch.from_numpy(labels).float(), labels.shape[1], texts.shape[1], labels.shape[0], mean_image, mean_text


def generate_test_database_ds(images, texts, labels):
    images = normalize(images)
    images = images - mean_image
    datasets = ImgFile(images, texts, labels, np.ones([labels.shape[0], 1]), np.ones([labels.shape[0], 1]))
    data_loader = data.DataLoader(dataset=datasets, batch_size=args.batch_size, shuffle=False, num_workers=5)
    return data_loader, torch.from_numpy(labels).float()


def evaluate():
    # Set the model to testing mode
    label_model.eval()
    autoencoder_gcn_model.eval()
    fusion_model.eval()

    database_codes = []
    for image, text, _, _, _ in database_loader:
        image = image.cuda()
        text = text.cuda()
        _, h_fusion = fusion_model(torch.cat((args.gamma * image, text), 1))
        codes = torch.sign(h_fusion)
        database_codes.append(codes.data.cpu().numpy())
    database_codes = np.concatenate(database_codes)

    test_codes = []
    for image, text, _, _, _ in test_loader:
        image = image.cuda()
        text = text.cuda()
        _, h_fusion = fusion_model(torch.cat((args.gamma * image, text), 1))
        codes = torch.sign(h_fusion)
        test_codes.append(codes.data.cpu().numpy())
    test_codes = np.concatenate(test_codes)

    map_500 = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'), 500)
    print(f'mAP@500: {map_500}')
    map_all = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'))
    print(f'mAP@All: {map_all}')
    global map_max
    if map_all > map_max:
        map_max = map_all
        sio.savemat('/data/yfchen/multi-modality_hashing/multi-modality_method_comparison/' + str(args.dataset) + '/MMH/' + str(args.bit) + 'bits/hash_codes.mat', {'B_te': test_codes, 'B_db': database_codes, 'L_te': test_labels.numpy(), 'L_db': database_labels.numpy()})
        # sio.savemat('c/' + str(args.c) + '_' + str(t+1) + '.mat', {'map_500': map_500, 'map_all': map_all})
        if args.nc == 2000:
            sio.savemat('result21/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 4000:
            sio.savemat('result22/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 6000:
            sio.savemat('result23/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        elif args.nc == 8000:
            sio.savemat('result24/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})
        else:
            sio.savemat('result25/' + str(t+1) + '/bit_' + str(args.bit) + '.mat', {'map_500': map_500, 'map_all': map_all})

    # database_codes = []
    # for image, _, _, _, _ in database_loader:
    #     image = image.cuda()
    #     text = torch.tile(torch.from_numpy(mean_text), (image.size(0), 1)).cuda()
    #     _, h_fusion = fusion_model(torch.cat((image, text), 1))
    #     codes = torch.sign(h_fusion)
    #     database_codes.append(codes.data.cpu().numpy())
    # database_codes = np.concatenate(database_codes)
    #
    # test_codes = []
    # for image, _, _, _, _ in test_loader:
    #     image = image.cuda()
    #     text = torch.tile(torch.from_numpy(mean_text), (image.size(0), 1)).cuda()
    #     _, h_fusion = fusion_model(torch.cat((image, text), 1))
    #     codes = torch.sign(h_fusion)
    #     test_codes.append(codes.data.cpu().numpy())
    # test_codes = np.concatenate(test_codes)
    #
    # map_500_i = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'), 500)
    # print(f'image to image mAP@500: {map_500_i}')
    # map_all_i = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'))
    # print(f'image to image mAP@All: {map_all_i}')
    # global map_max_i
    # if map_all_i > map_max_i:
    #     map_max_i = map_all_i
    #     sio.savemat('/data/yfchen/multi-modality_hashing/multi-modality_method_comparison/' + str(args.dataset) + '/MMH622/' + str(args.bit) + 'bits/hash_codes_i.mat', {'B_te': test_codes, 'B_db': database_codes, 'L_te': test_labels.numpy(), 'L_db': database_labels.numpy()})
    #     if args.nc == 2000:
    #         sio.savemat('result21/' + str(t+1) + '/bit_' + str(args.bit) + '_i.mat', {'map_500': map_500_i, 'map_all': map_all_i})
    #     elif args.nc == 4000:
    #         sio.savemat('result22/' + str(t+1) + '/bit_' + str(args.bit) + '_i.mat', {'map_500': map_500_i, 'map_all': map_all_i})
    #     elif args.nc == 6000:
    #         sio.savemat('result23/' + str(t+1) + '/bit_' + str(args.bit) + '_i.mat', {'map_500': map_500_i, 'map_all': map_all_i})
    #     elif args.nc == 8000:
    #         sio.savemat('result24/' + str(t+1) + '/bit_' + str(args.bit) + '_i.mat', {'map_500': map_500_i, 'map_all': map_all_i})
    #     else:
    #         sio.savemat('result25/' + str(t+1) + '/bit_' + str(args.bit) + '_i.mat', {'map_500': map_500_i, 'map_all': map_all_i})
    #
    # database_codes = []
    # for _, text, _, _, _ in database_loader:
    #     image = torch.zeros([text.size(0), 4096], dtype=torch.float).cuda()
    #     text = text.cuda()
    #     _, h_fusion = fusion_model(torch.cat((image, text), 1))
    #     codes = torch.sign(h_fusion)
    #     database_codes.append(codes.data.cpu().numpy())
    # database_codes = np.concatenate(database_codes)
    #
    # test_codes = []
    # for _, text, _, _, _ in test_loader:
    #     image = torch.zeros([text.size(0), 4096], dtype=torch.float).cuda()
    #     text = text.cuda()
    #     _, h_fusion = fusion_model(torch.cat((image, text), 1))
    #     codes = torch.sign(h_fusion)
    #     test_codes.append(codes.data.cpu().numpy())
    # test_codes = np.concatenate(test_codes)
    #
    # map_500_t = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'), 500)
    # print(f'text to text mAP@500: {map_500_t}')
    # map_all_t = mean_average_precision(torch.from_numpy(test_codes), torch.from_numpy(database_codes), test_labels, database_labels, torch.device('cuda'))
    # print(f'text to text mAP@All: {map_all_t}')
    # global map_max_t
    # if map_all_t > map_max_t:
    #     map_max_t = map_all_t
    #     sio.savemat('/data/yfchen/multi-modality_hashing/multi-modality_method_comparison/' + str(args.dataset) + '/MMH622/' + str(args.bit) + 'bits/hash_codes_t.mat', {'B_te': test_codes, 'B_db': database_codes, 'L_te': test_labels.numpy(), 'L_db': database_labels.numpy()})
    #     if args.nc == 2000:
    #         sio.savemat('result21/' + str(t+1) + '/bit_' + str(args.bit) + '_t.mat', {'map_500': map_500_t, 'map_all': map_all_t})
    #     elif args.nc == 4000:
    #         sio.savemat('result22/' + str(t+1) + '/bit_' + str(args.bit) + '_t.mat', {'map_500': map_500_t, 'map_all': map_all_t})
    #     elif args.nc == 6000:
    #         sio.savemat('result23/' + str(t+1) + '/bit_' + str(args.bit) + '_t.mat', {'map_500': map_500_t, 'map_all': map_all_t})
    #     elif args.nc == 8000:
    #         sio.savemat('result24/' + str(t+1) + '/bit_' + str(args.bit) + '_t.mat', {'map_500': map_500_t, 'map_all': map_all_t})
    #     else:
    #         sio.savemat('result25/' + str(t+1) + '/bit_' + str(args.bit) + '_t.mat', {'map_500': map_500_t, 'map_all': map_all_t})


if __name__ == '__main__':
    args = parse_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if args.dataset == 'mirflickr':
        IAll = sio.loadmat('/data/yfchen/dataset/MIR-Flickr25K/cleared_image_vgg19.mat')['cleared_image_vgg19'].astype(np.float32)
        TAll = sio.loadmat('/data/yfchen/dataset/MIR-Flickr25K/matlab/cleared_image_BoW.mat')['cleared_image_BoW'].astype(np.float32)
        LAll = sio.loadmat('/data/yfchen/dataset/MIR-Flickr25K/matlab/cleared_image_annotations_24.mat')['cleared_image_annotations_24'].astype(np.float32)
        index = sio.loadmat('/data/yfchen/multi-modality_hashing/MMH/data/mirflickr/index.mat')
    elif args.dataset == 'nuswide':
        IAll = sio.loadmat('/data/yfchen/dataset/NUS_WIDE/nuswide_vgg19.mat')['nuswide_vgg19'].astype(np.float32)
        TAll = sio.loadmat('/data/yfchen/dataset/NUS_WIDE/nus-wide-tc21-yall.mat')['YAll'].astype(np.float32)
        LAll = sio.loadmat('/data/yfchen/dataset/NUS_WIDE/nus-wide-tc21-lall.mat')['LAll'].astype(np.float32)
        index = sio.loadmat('/data/yfchen/multi-modality_hashing/MMH/data/nuswide/index.mat')
    elif args.dataset == 'mscoco':
        IAll = sio.loadmat('/data1/yfchen/MS-COCO/mscoco_vgg19.mat')['mscoco_vgg19'].astype(np.float32)
        TAll = sio.loadmat('/data1/yfchen/MS-COCO/ms-coco-yall.mat')['YAll'].astype(np.float32)
        LAll = sio.loadmat('/data1/yfchen/MS-COCO/ms-coco-lall.mat')['LAll'].astype(np.float32)
        index = sio.loadmat('/data/yfchen/multi-modality_hashing/MMH/data/mscoco/index.mat')
    else:
        print("This dataset does not exist!")
    indQ = index['indQ'].squeeze()
    indT = index['indT'].squeeze()
    indD = index['indD'].squeeze()
    # the query dataset
    I_te = IAll[indQ]
    T_te = TAll[indQ]
    L_te = LAll[indQ]
    # the train dataset
    I_tr = IAll[indT]
    T_tr = TAll[indT]
    L_tr = LAll[indT]
    # the retrieval dataset
    I_db = IAll[indD]
    T_db = TAll[indD]
    L_db = LAll[indD]
    del IAll, TAll, LAll

    BCELoss = torch.nn.BCELoss()

    for t in range(args.times):

        train_loader, train_labels, label_dim, text_dim, num_train, mean_image, mean_text = generate_train_ds(I_tr, T_tr, L_tr)
        train_labels = train_labels.cuda()
        test_loader, test_labels = generate_test_database_ds(I_te, T_te, L_te)
        database_loader, database_labels = generate_test_database_ds(I_db, T_db, L_db)
        print('Data loader has been generated!Image dimension = 4096, text dimension = %d, label dimension = %d.' % (text_dim, label_dim))

        for args.bit in [32]:
            print('nc = %d' % args.nc)
            print('gamma = %f' % args.gamma)
            print('lamda = %f' % args.lamda)
            print('a = %f' % args.a)
            print('p1 = %f' % args.p1)
            print('p2 = %f' % args.p2)
            print('bit = %d' % args.bit)

            label_model = LabelModule(label_dim, args.bit)
            label_model.cuda()

            autoencoder_gcn_model = AutoEncoderGcnModule(4096, text_dim)
            autoencoder_gcn_model.cuda()

            fusion_model = FusionModule(4096 + text_dim, args.bit)
            fusion_model.cuda()

            lr_l = 0.1
            lr_a = 0.1
            lr_f = 0.1
            # lr_decay = np.exp(np.linspace(0, -8, args.Epoch_num))
            lr_decay = np.linspace(1, 0.01, args.Epoch_num)

            map_max = 0
            map_max_i = 0
            map_max_t = 0
            Losses = []
            for Epoch in range(args.Epoch_num):
                print('Epoch: %d' % (Epoch + 1))

                # Set the model to training mode
                label_model.train()
                autoencoder_gcn_model.train()
                fusion_model.train()

                # set the optimizer
                label_optimizer = torch.optim.Adam(label_model.parameters(), lr=lr_l * lr_decay[Epoch])
                autoencoder_gcn_optimizer = torch.optim.Adam(autoencoder_gcn_model.parameters(), lr=lr_a * lr_decay[Epoch])
                fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr_f * lr_decay[Epoch])

                # features of label modality
                for epoch in range(5):
                    for i in range(20):
                        s = calculate_s(train_labels, train_labels)
                        all_h_label = label_model(train_labels)
                        loss_label = negative_log_likelihood_similarity_loss1(all_h_label, all_h_label.data, s, args.bit) \
                            + args.beta * quantization_loss1(all_h_label)
                        label_optimizer.zero_grad()
                        loss_label.backward()
                        label_optimizer.step()
                    print('Loss label: %.4f' % loss_label.data.cpu().numpy())

                # autoencoder
                for epoch in range(5):
                    Loss1 = 0
                    Loss2 = 0
                    for i, (image, text, label, m1, m2) in enumerate(train_loader):
                        label = label.cuda()
                        image = image.cuda()
                        text = text.cuda()
                        m1 = m1.cuda()
                        m2 = m2.cuda()
                        # construct the graph for this batch
                        graph = torch.mm(label, label.T)
                        m = args.c * torch.mul(m1, m2).T.repeat(500, 1) + (1 - args.c) * (1 - torch.mul(m1, m2).T.repeat(500, 1))
                        graph = torch.mul(graph, m)
                        g = graph - torch.mul(graph, torch.eye(label.size(0)).cuda())
                        p = torch.mul(m1, m2) * args.p1 + (1 - torch.mul(m1, m2)) * args.p2
                        graph = (1 - p) * (g / torch.sum(g + 1e-6, dim=1, keepdim=True) + torch.diag_embed((torch.sum(g, dim=1) < 0.5).float())) + p * torch.eye(label.size(0)).cuda()
                        input_image = deepcopy(image)
                        input_text = deepcopy(text)
                        output_image, output_text, _ = autoencoder_gcn_model(graph, torch.cat((args.gamma * input_image, input_text), 1))
                        loss1 = args.lamda * torch.mean(torch.mul(output_image - args.gamma * image, m1) ** 2)
                        loss2 = BCELoss(torch.mul(output_text, m2), torch.mul(text, m2))
                        loss = loss1 + loss2
                        Loss1 += loss1.data.cpu().numpy()
                        Loss2 += loss2.data.cpu().numpy()
                        autoencoder_gcn_optimizer.zero_grad()
                        loss.backward()
                        autoencoder_gcn_optimizer.step()
                    print('Loss antoencoder image: %.4f, Loss antoencoder text: %.4f' % (Loss1 / i, Loss2 / i))

                for epoch in range(5):
                    Loss1 = 0
                    Loss2 = 0
                    Loss3 = 0
                    for i, (image, text, label, m1, m2) in enumerate(train_loader):
                        label = label.cuda()
                        image = image.cuda()
                        text = text.cuda()
                        m1 = m1.cuda()
                        m2 = m2.cuda()
                        s = calculate_s(label, train_labels)
                        # construct the graph for this batch
                        graph = torch.mm(label, label.T)
                        m = args.c * torch.mul(m1, m2).T.repeat(500, 1) + (1 - args.c) * (1 - torch.mul(m1, m2).T.repeat(500, 1))
                        graph = torch.mul(graph, m)
                        g = graph - torch.mul(graph, torch.eye(label.size(0)).cuda())
                        p = torch.mul(m1, m2) * args.p1 + (1 - torch.mul(m1, m2)) * args.p2
                        graph = (1 - p) * (g / torch.sum(g + 1e-6, dim=1, keepdim=True) + torch.diag_embed((torch.sum(g, dim=1) < 0.5).float())) + p * torch.eye(label.size(0)).cuda()
                        _, _, latent = autoencoder_gcn_model(graph, torch.cat((args.gamma * image, text), 1))
                        fusion, h_fusion = fusion_model(torch.cat((args.gamma * image, text), 1))
                        loss1 = args.a * correspondence_loss(fusion, latent)
                        loss2 = args.alpha * negative_log_likelihood_similarity_loss1(h_fusion, all_h_label.data, s, args.bit)
                        loss3 = args.beta * quantization_loss1(h_fusion)
                        loss = loss1 + loss2 + loss3
                        Loss1 += loss1.data.cpu().numpy()
                        Loss2 += loss2.data.cpu().numpy()
                        Loss3 += loss3.data.cpu().numpy()
                        fusion_optimizer.zero_grad()
                        loss.backward()
                        fusion_optimizer.step()
                    print('Latent Loss: %.4f, Similarity Loss: %.4f, Quantization Loss: %.10f' % (Loss1 / i, Loss2 / i, Loss3 / i))

                # Losses.append(Loss1 / i + Loss2 / i + Loss3 / i)
                evaluate()
            # sio.savemat('Losses_mscoco_128.mat', {'Losses': Losses})
