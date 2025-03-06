import os
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from data import x3_gap_toy, sin_toy
from toy import polynomial_toy, polynomial_gap_toy, polynomial_gap_toy2, g2_toy, g3_toy, isotropy_toy
from bnnp import BNNMC, BNNGZ
from bnn import BNN
from gp_prior import ExactGPModel, prior_sample_functions, prior_sample_functions2
import gpytorch
from utils.spectral import SpectralScoreEstimator

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils.utils import default_plotting_new as init_plotting


################################### Hyper-parameters ###################################

lr_bnn = 0.001
prior_coeff = 1
bnn_name_string = 'VIMC'   # mcmc_response
random_seed = 123
max_epoch_num = 100
save_model_freq = 5000
log_model_freq = 3000
test_interval = 2000
num_sample = 1000
epochs = 10001
n_step_prior_pretraining = 100
regu_coeff = 0.01

torch.manual_seed(random_seed)
np.random.seed(random_seed)


################################### Network Architecture ###################################

n_units = 100
n_hidden = 2
hidden_dims = [n_units] * n_hidden
activation_fn = 'tanh'


print("============================================================================================")
################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")
###################### logging ######################

#### log files for multiple runs are NOT overwritten

log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + '/' + bnn_name_string + '/' + 'fsgld_toy'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/' + bnn_name_string + 'fsgld_toy' + "_" + str(run_num) + ".csv"

print("current logging run number for " + bnn_name_string + " : ", run_num)
print("logging at : " + log_f_name)


print("============================================================================================")
################### checkpointing ###################

run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

directory = "./pretrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + bnn_name_string + '/' + 'fsgld_toy'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "{}_{}_{}.pth".format(bnn_name_string, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

################### savefigures ###################

results_folder = "./results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

figures_folder = results_folder + '/' + bnn_name_string + '/' + 'fsgld_toy'
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)


print("============================================================================================")
############# print all hyperparameters #############

print("learning rate: ", lr_bnn)
print("coefficient of prior regularization: ", prior_coeff)
print("random seed: ", random_seed)
print("max number of epoches: ", max_epoch_num)


print("============================================================================================")
############################## load and normalize data ##############################

dataset = dict(x3=x3_gap_toy, sin=sin_toy, g3=g3_toy, g2=g2_toy, gap=polynomial_gap_toy, iso=isotropy_toy)['g3']()
original_x_train, original_y_train = dataset.train_samples()
mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)
mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
train_x = (original_x_train - mean_x) / std_x
train_y = (original_y_train - mean_y) / std_y
original_x_test, original_y_test = dataset.test_samples()
test_x = (original_x_test - mean_x) / std_x
test_y = (original_y_test - mean_y) / std_y

y_logstd = np.log(dataset.y_std / std_y)

lower_ap = (dataset.x_min - mean_x) / std_x
upper_ap = (dataset.x_max - mean_x) / std_x

training_num, input_dim = np.shape(train_x)
input_dim = 1 #np.array([input_dim])
output_dim = 1 #np.array([1])
is_continuous = True

original_x_train = torch.from_numpy(original_x_train).float().to(device)
original_y_train = torch.from_numpy(original_y_train).float().to(device)
original_x_test = torch.from_numpy(original_x_test).float().to(device)
original_y_test = torch.from_numpy(original_y_test).float().to(device)

add_set = np. random. uniform(-5, 5, (40, 1))
add_set_y = 2 * np. sin(4*add_set)
add_set = torch.from_numpy(add_set).float().to(device)
add_set_y = torch.from_numpy(add_set_y).float().to(device)
add_set_y = torch.squeeze(add_set_y).to(device)

original_x_train_add = torch.cat((original_x_train, add_set), 0).to(device)
original_y_train_add = torch.cat((original_y_train, add_set_y), 0).to(device)

# add_set_gap = np. random. uniform(-7.5, 7.5, (40, 1))
# add_set_y_gap = np.sin(add_set_gap) + 0.1 * add_set_gap ** 2
# add_set_gap = torch.from_numpy(add_set_gap).float().to(device)
# add_set_y_gap = torch.from_numpy(add_set_y_gap).float().to(device)
# add_set_y_gap = torch.squeeze(add_set_y_gap).to(device)
#
# original_x_train_add_gap = torch.cat((original_x_train, add_set_gap), 0).to(device)
# original_y_train_add_gap = torch.cat((original_y_train, add_set_y_gap), 0).to(device)

add_set_gap = np. random. uniform(-10, 10, (40, 1))
add_set_y_gap = np.sin(add_set_gap) + 0.1 * add_set_gap ** 2
add_set_gap = torch.from_numpy(add_set_gap).float().to(device)
add_set_y_gap = torch.from_numpy(add_set_y_gap).float().to(device)
add_set_y_gap = torch.squeeze(add_set_y_gap).to(device)

original_x_train_add_gap = torch.cat((original_x_train, add_set_gap), 0).to(device)
original_y_train_add_gap = torch.cat((original_y_train, add_set_y_gap), 0).to(device)

#####g2########
add_set_g2 = np. random. uniform(-10, 10, (30, 1))
add_set_y_g2 = np.sin(add_set_g2) + 0.1 * add_set_g2
add_set_g2 = torch.from_numpy(add_set_g2).float().to(device)
add_set_y_g2 = torch.from_numpy(add_set_y_g2).float().to(device)
add_set_y_g2 = torch.squeeze(add_set_y_g2).to(device)

original_x_train_add_g2 = torch.cat((original_x_train, add_set_g2), 0).to(device)
original_y_train_add_g2 = torch.cat((original_y_train, add_set_y_g2), 0).to(device)

#####g3
add_set_g3 = np. random. uniform(-1, 1, (40, 1))    # 40
add_set_y_g3 = np.sin(3 * 3.14 * add_set_g3) + 0.3 * np.cos(9 * 3.14 * add_set_g3) + 0.5 * np.sin(7 * 3.14 * add_set_g3)
add_set_g3 = torch.from_numpy(add_set_g3).float().to(device)
add_set_y_g3 = torch.from_numpy(add_set_y_g3).float().to(device)
add_set_y_g3 = torch.squeeze(add_set_y_g3).to(device)

original_x_train_add_g3 = torch.cat((original_x_train, add_set_g3), 0).to(device)
original_y_train_add_g3 = torch.cat((original_y_train, add_set_y_g3), 0).to(device)

###### iso
add_set_iso = np. random. uniform(-0.5, 1.2, (40, 1))
add_set_y_iso = add_set_iso + 0.3 * np.sin(2 * 3.14 * add_set_iso) + 0.3 * np.sin(4 * 3.14 * add_set_iso)
add_set_iso = torch.from_numpy(add_set_iso).float().to(device)
add_set_y_iso = torch.from_numpy(add_set_y_iso).float().to(device)
add_set_y_iso = torch.squeeze(add_set_y_iso).to(device)

original_x_train_add_iso = torch.cat((original_x_train, add_set_iso), 0).to(device)
original_y_train_add_iso = torch.cat((original_y_train, add_set_y_iso), 0).to(device)

#####g3_add_except_training data
add_1 = np. random. uniform(-1, -0.75, (20, 1))
add_2 = np. random. uniform(-0.5, 0.25, (20, 1))
add_3 = np. random. uniform(0.75, 1, (20, 1))
add = np.concatenate((add_1, add_2, add_3), axis=0)

add_g3_y = np.sin(3 * 3.14 * add) + 0.3 * np.cos(9 * 3.14 * add) + 0.5 * np.sin(7 * 3.14 * add)
add_g3_y = torch.from_numpy(add_g3_y).float().to(device)
add_g3_y = torch.squeeze(add_g3_y).to(device)

add_g3 = torch.from_numpy(add).float().to(device)
print('add_g3.shape: ', add_g3.shape)

loss = nn.MSELoss(reduction='mean')
print("============================================================================================")
################################## define model ##################################

bnn = BNNMC(input_dim, output_dim, hidden_dims, activation_fn, is_continuous, scaled_variance=True).to(device)

# bnn_optimizer = torch.optim.Adam([{'params': bnn.parameters(), 'lr': lr_bnn}])
# bnn_optimizer = torch.optim.SGD(bnn.parameters(), lr=lr_bnn, weight_decay=5e-6)
bnn_optimizer = torch.optim.Adam(bnn.parameters(), lr=lr_bnn)


print("============================================================================================")
################################## start training ##################################
################################## prior pre-training ##################################

# pre-train GP prior

likelihood = gpytorch.likelihoods.GaussianLikelihood()
prior = ExactGPModel(original_x_train, original_y_train, likelihood, input_dim).to(device)

prior.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(prior.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, prior)

gp_start_time = datetime.now().replace(microsecond=0)
print("GP Started training at (GMT) : ", gp_start_time)

for i in range(n_step_prior_pretraining):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = prior(original_x_train)
    # Calc loss and backprop gradients
    loss_gp = -mll(output, original_y_train)
    loss_gp.backward()
    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #     i + 1, n_step_prior_pretraining, loss_gp.item(),
    #     prior.covar_module.base_kernel.lengthscale.item(),
    #     prior.likelihood.noise.item()
    # ))
    print('Iter %d/%d - Loss: %.3f     noise: %.3f' % (
        i + 1, n_step_prior_pretraining, loss_gp.item(),
        # prior.covar_module.base_kernel.lengthscale.item(),
        prior.likelihood.noise.item()
    ))

    optimizer.step()

gp_end_time = datetime.now().replace(microsecond=0)
print("GP Finished training at (GMT) : ", gp_end_time)
print("Total pre-training time  : ", gp_end_time - gp_start_time)

prior.eval()
likelihood.eval()

with torch.no_grad():
    plt.clf()
    figure = plt.figure(figsize=(8, 5.5), facecolor='white')
    init_plotting()
    p_gp = likelihood(prior(original_x_test))
    f_var = prior(original_x_test).variance
    f_std = f_var ** 0.5
    plt.plot(original_x_test.squeeze().cpu(), original_y_test.cpu(), 'g', label="True function")
    plt.plot(original_x_test.squeeze().cpu(), p_gp.mean.cpu().numpy(), 'slateblue', label='GP posterior', linewidth=2.5)
    for i in range(5):
        plt.fill_between(original_x_test.squeeze().cpu(), p_gp.mean.cpu().numpy() - i * 1 * p_gp.variance.cpu().numpy(),
                         p_gp.mean.cpu().numpy() - (i + 1) * 1 * p_gp.variance.cpu().numpy(), linewidth=0.0,
                         alpha=1.0 - i * 0.15, color='orchid')  # wheat
        plt.fill_between(original_x_test.squeeze().cpu(), p_gp.mean.cpu().numpy() + i * 1 * p_gp.variance.cpu().numpy(),
                         p_gp.mean.cpu().numpy() + (i + 1) * 1 * p_gp.variance.cpu().numpy(), linewidth=0.0,
                         alpha=1.0 - i * 0.15, color='orchid')
    plt.plot(original_x_train.cpu(), original_y_train.cpu(), 'ok', zorder=10, ms=6, label='Observations')
    #################
    # for i in range(5):
    #     plt.fill_between(original_x_test.squeeze().cpu(), p_gp.mean.cpu().numpy() - i * 1 * f_std.cpu().numpy(),
    #                      p_gp.mean.cpu().numpy() - (i + 1) * 1 * f_std.cpu().numpy(), linewidth=0.0,
    #                      alpha=1.0 - i * 0.15, color='orchid')  # wheat
    #     plt.fill_between(original_x_test.squeeze().cpu(), p_gp.mean.cpu().numpy() + i * 1 * f_std.cpu().numpy(),
    #                      p_gp.mean.cpu().numpy() + (i + 1) * 1 * f_std.cpu().numpy(), linewidth=0.0,
    #                      alpha=1.0 - i * 0.15, color='orchid')
    # plt.plot(original_x_train.cpu(), original_y_train.cpu(), 'ok', zorder=10, ms=6, label='Observations')

    ######g3######
    plt.axvline(-0.75, linestyle='--', color='darkorchid')
    plt.axvline(-0.25, linestyle='--', color='darkorchid')
    plt.axvline(0.25, linestyle='--', color='darkorchid')
    plt.axvline(0.75, linestyle='--', color='darkorchid')
    plt.text(-0.75, -1.5, 'Training data range', fontsize='small')
    plt.text(0.25, 1, 'Training data range', fontsize='small')

    plt.legend(loc='upper center', ncol=3, fontsize='small')
    plt.title('GP posteior')
    plt.tight_layout()
    plt.ylim([dataset.y_min, dataset.y_max])
    plt.tight_layout()
    plt.savefig(figures_folder + '/gp posterior.pdf')

with torch.no_grad():

    plt.clf()
    figure = plt.figure(figsize=(8, 5.5), facecolor='white')
    init_plotting()

    # measurement_set = sample_measurement_set(X=original_x_train_add, num_data=60)
    gpss = prior_sample_functions(original_x_test, prior, num_sample=128).detach().float().to(device)
    # gpss = prior_bnn.sample_functions(original_x_test, num_sample=128).float().to(device)
    gpss = gpss.squeeze()
    gpss = torch.mean(gpss, 0)
    plt.plot(original_x_test.squeeze().cpu(), original_y_test.cpu(), 'g', label="True function")
    plt.plot(original_x_test.squeeze().cpu(), gpss.cpu().numpy(), 'slateblue', label='GP prior', linewidth=2.5)
    plt.legend(loc='upper center', ncol=3, fontsize='small')
    plt.title('GP prior')
    plt.tight_layout()
    plt.ylim([dataset.y_min, dataset.y_max])
    plt.tight_layout()
    plt.savefig(figures_folder + '/gp prior.pdf')

with torch.no_grad():
    gpss = prior_sample_functions(original_x_train, prior, num_sample=128).detach().float().to(device)
    # gpss = prior_bnn.sample_functions(original_x_test, num_sample=128).float().to(device)
    gpss = gpss.squeeze()
    prior_mean = torch.mean(gpss, 0)
    prior_var = torch.var(gpss, 0)
    print('mean.shape:', prior_mean)
    print('var.shape: ', prior_var)


# with torch.no_grad():
#     fprior = likelihood(prior(original_x_train))
#     prior_mean = fprior.mean
#     prior_var = fprior.variance
#     print('mean.shape:', prior_mean.shape)
#     print('var.shape: ', prior_var.shape)

# ###################### define noise_loss ###############################
def noise_loss(lr):
    noise_loss = 0.0
    noise_std = (2/lr)**0.5
    for var in bnn.parameters():
        means = torch.zeros(var.size()).to(device)
        sigma = torch.normal(means, std=noise_std).to(device)
        noise_loss += torch.sum(var * sigma)
    return noise_loss

def updata_params(lr, epoch):
    for p in bnn.parameters():
        if not hasattr(p, 'buf'):
            p.buf = torch.zeros(p.size()).to(device)
        d_p = p.grad.data
        d_p.add_(weight_decay, p.data)
        buf_new = (1-0.9) * p.buf - lr * d_p
        if epoch > 8000:
            eps = torch.randn(p.size()).to(device)
            buf_new += (2.0 * lr * 1e-6) ** 0.5 * eps
        p.data.add_(buf_new)
        p.buf = buf_new

def sample_measurement_set(X, num_data):
    n = torch.Tensor([40])  # sin_40, gap_20
    # sample measurement set with size n
    perm = torch.randperm(int(num_data))
    idx = perm[:n.to(torch.long)]
    measurement_set = X[idx, :]
    return measurement_set
#
# ################################## start training ##################################
mt = 0
weight_decay = 5e-6
# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

# logging file
log_f = open(log_f_name, "w+")
log_f.write('epoch,training_loss\n')
scheduler = torch.optim.lr_scheduler.StepLR(bnn_optimizer, 5000, gamma=0.9, last_epoch=-1)
train_loss_all = []
########### SGLD ###############
for epoch in range(epochs):

    # loss

    pred_y = bnn.forward(original_x_train)
    # y_predicted, distance_prior = bnn.forward_w(train_x)

    pred_y = pred_y.squeeze().flatten()

    ##################### training data as measurement set for d_logp
    # d_logp = ((pred_y - prior_mean) / prior_var).detach()
    # d_logp = ((pred_y - prior_mean)).detach()
    # d_logp = pred_y * d_logp
    # d_logp = torch.mean(d_logp)
    # print('d_logp.shape: ', d_logp.shape)

    ###################  measurement set for d_logp
    m_set = sample_measurement_set(X=original_x_train_add_g3, num_data=60)
    y_m_set = bnn.forward(m_set).squeeze().flatten()
    # target_mset = torch.sin(3 * 3.14 * m_set) + 0.3 * torch.cos(9 * 3.14 * m_set) + 0.5 * torch.sin(7 * 3.14 * m_set)
    # target_mset = target_mset.squeeze().flatten()
    # d_like = y_m_set * ((y_m_set - target_mset).detach())
    # d_like = torch.mean(d_like)

    with torch.no_grad():
        fprior_m = likelihood(prior(m_set))
        prior_mean_m = fprior_m.mean
        prior_var_m = fprior_m.variance
        # print('var_m: ', prior_var_m)
    # # prior_mean_m = torch.randn_like(y_m_set)
    # prior_mean_m = torch.zeros_like(y_m_set)
    # prior_var_m = torch.ones_like(y_m_set)
    # # prior_var_m = torch.randn_like(y_m_set)

    #
        # fprior_t = likelihood(prior(original_x_train))
        # prior_mean_t = fprior_t.mean
        # prior_var_t = fprior_t.variance
    #     # print('var_t: ', prior_var_t)
    #
    d_logp_m = ((y_m_set - prior_mean_m) / prior_var_m).detach()
    # print('d_log_m:', d_logp_m)
    d_logp_m = y_m_set * d_logp_m
    d_logp_m = torch.mean(d_logp_m)
    # print('d_log_m:', d_logp_m)
    #
    # d_logp_t = ((pred_y - prior_mean_t) / prior_var_t).detach()
    # # d_logp_t = ((pred_y - prior_mean_t)).detach()
    # d_logp_t = pred_y * d_logp_t
    # d_logp_t = torch.mean(d_logp_t)


    # m_like = bnn.forward(add_g3).squeeze().flatten()
    # m_like = m_like * ((m_like - add_g3_y).detach())
    # m_like = torch.mean(m_like)
    t_like = pred_y * ((pred_y - original_y_train).detach())
    t_like = torch.mean(t_like)

    ####################
    train_loss = t_like + 1e-3 * d_logp_m  # 1e-6  1e-2   1e-3
    train_loss_all.append(train_loss.item())

    # train_loss = loss(pred_y, original_y_train) + 1e-6 * d_logp  # coeff: 1e-6 1e-3
    if epoch > 7000:           # 7000
        noise = noise_loss(lr_bnn)
        train_loss = train_loss + 1e-6 * noise

    # optimisation
    bnn_optimizer.zero_grad()
    train_loss.backward()
    bnn_optimizer.step()
    scheduler.step()

########### SGHMC ##########
# lr = lr_bnn
# for epoch in range(epochs):
#
#     pred_y = bnn.forward(original_x_train)
#     pred_y = pred_y.squeeze().flatten()
#     # train_loss = loss(pred_y, original_y_train)
#
#     t_like = pred_y * ((pred_y - original_y_train).detach())
#     t_like = torch.mean(t_like)
#     train_loss = t_like
#
#     m_set = sample_measurement_set(X=original_x_train_add_g3, num_data=60)
#     y_m_set = bnn.forward(m_set).squeeze().flatten()
#
#     with torch.no_grad():
#         fprior_m = likelihood(prior(m_set))
#         prior_mean_m = fprior_m.mean
#         prior_var_m = fprior_m.variance
#
#     # prior_mean_m = torch.randn_like(y_m_set)
#     # # prior_mean_m = torch.zeros_like(y_m_set)
#     # prior_var_m = torch.ones_like(y_m_set)
#     # prior_var_m = torch.randn_like(y_m_set)
#
#         # fprior_t = likelihood(prior(original_x_train))
#         # prior_mean_t = fprior_t.mean
#         # prior_var_t = fprior_t.variance
#
#
#     d_logp_m = ((y_m_set - prior_mean_m) / prior_var_m).detach()
#     d_logp_m = y_m_set * d_logp_m
#     d_logp_m = torch.mean(d_logp_m)
#     #
#     # d_logp_t = ((pred_y - prior_mean_t) / prior_var_t).detach()
#     # # d_logp_t = ((pred_y - prior_mean_t)).detach()
#     # d_logp_t = pred_y * d_logp_t
#     # d_logp_t = torch.mean(d_logp_t)
#
#     train_loss = t_like + 1e-2 * d_logp_m  # 1e-6  1e-2
#     train_loss_all.append(train_loss.item())
#
#     bnn.zero_grad()
#     train_loss.backward()
#     # updata_params(lr_bnn, epoch)
#
#     ##################### g(. z) = z ###############
#     for p in bnn.parameters():
#         # if not hasattr(p, 'buf'):
#         #     p.buf = torch.zeros(p.size()).to(device)
#         if not hasattr(p, 'm'):
#             p.m = torch.zeros(p.size()).to(device)
#         if not hasattr(p, 'v'):
#             p.v = torch.zeros(p.size()).to(device)
#
#         d_p = p.grad.data
#         p.m = 0.9 * p.m + (1 - 0.9) * d_p
#         p.v = 0.999 * p.v + (1 - 0.999) * (d_p ** 2)
#         m_ = p.m / (1 - (0.9 ** (epoch + 1)))
#         v_ = p.v / (1 - (0.999 ** (epoch + 1)))
#         d_p = lr * m_ / (v_ ** 0.5 + 1e-8)
#         # p.data = p.data - d_p
#         p.buf = 1e-2 * torch.randn(p.size()).to(device)
#         buf_new = p.buf - d_p  # 1e-2 * p.buf - d_p
#
#         p.data = p.data + buf_new
#         p.buf = buf_new

    ####### g(. z) is same structure bnn as f(. w) #############
    # z_list = []
    # for p in bnn.parameters():
    #     # if not hasattr(p, 'buf'):
    #     #     p.buf = torch.zeros(p.size()).to(device)
    #     if not hasattr(p, 'm'):
    #         p.m = torch.zeros(p.size()).to(device)
    #     if not hasattr(p, 'v'):
    #         p.v = torch.zeros(p.size()).to(device)
    #
    #     d_p = p.grad.data
    #     p.m = 0.9 * p.m + (1 - 0.9) * d_p
    #     p.v = 0.999 * p.v + (1 - 0.999) * (d_p ** 2)
    #     m_ = p.m / (1 - (0.9 ** (epoch + 1)))
    #     v_ = p.v / (1 - (0.999 ** (epoch + 1)))
    #     d_p = lr * m_ / (v_ ** 0.5 + 1e-8)
    #     # p.data = p.data - d_p
    #
    #     p.buf = 1e-1 * torch.randn(p.size()).to(device)
    #     buf_new = p.buf - d_p  # 1e-2 * p.buf - d_p
    #     z_list.append(buf_new)
    #
    # g = BNNGZ(z_list, input_dim, activation_fn, hidden_dims, norm_layer=None).to(device)
    #
    # # z_set = sample_measurement_set(X=original_x_train_add_g3, num_data=60)
    # g_mset = g.forward(m_set).squeeze().flatten()
    #
    # d_logp_g = ((g_mset - prior_mean_m) / prior_var_m).detach()
    # d_logp_g = g_mset * d_logp_g
    # d_logp_g = torch.mean(d_logp_g)
    #
    # g_loss = d_logp_g  # 1e-6
    #
    # g.zero_grad()
    # g_loss.backward()
    #
    # z_new_list = []
    # for p in g.parameters():
    #     if not hasattr(p, 'm'):
    #         p.m = torch.zeros(p.size()).to(device)
    #     if not hasattr(p, 'v'):
    #         p.v = torch.zeros(p.size()).to(device)
    #
    #     d_z = p.grad.data
    #     p.m = 0.9 * p.m + (1 - 0.9) * d_z
    #     p.v = 0.999 * p.v + (1 - 0.999) * (d_z ** 2)
    #     m_ = p.m / (1 - (0.9 ** (epoch + 1)))
    #     v_ = p.v / (1 - (0.999 ** (epoch + 1)))
    #     d_z = lr * m_ / (v_ ** 0.5 + 1e-8)
    #     z_new_list.append(d_z)
    #
    # i = 0
    # for p in bnn.parameters():
    #     p.data = p.data + z_new_list[i]
    #     i = i +1


    # print
    print("epoch : {} \t\t training loss \t\t : {}".format(epoch, train_loss),
          datetime.now().replace(microsecond=0) - start_time)

    # log in logging file
    if epoch % log_model_freq == 0:
        print("--------------------------------------------------------------------------------------------")
        log_f.write('{},{}\n'.format(epoch, train_loss))
        log_f.flush()
        print("log saved")
        print("--------------------------------------------------------------------------------------------")

    # save model weights
    if epoch % save_model_freq == 0:
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        bnn.save(checkpoint_path)
        print("model saved")
        print("--------------------------------------------------------------------------------------------")

    if epoch % test_interval == 0:

        bnn.eval()

        samples_pred_test_y = bnn.forward_eval(original_x_test, num_sample).squeeze().detach()

        # mean_y_pred = pred_test_y
        mean_y_pred = torch.mean(samples_pred_test_y, 0).cpu().numpy()
        std_y_pred = torch.std(samples_pred_test_y, 0).cpu().numpy()
        test_loss = loss(samples_pred_test_y, original_y_test)

        bnn.train()

        plt.clf()
        figure = plt.figure(figsize=(8, 5.5), facecolor='white')
        init_plotting()

        plt.plot(original_x_test.squeeze().cpu(), original_y_test.cpu(), 'g', label="True function")
        plt.plot(original_x_test.squeeze().cpu(), mean_y_pred, 'slateblue', label='Mean function', linewidth=2.5)  # chocolate
        for i in range(5):
            plt.fill_between(original_x_test.squeeze().cpu(), mean_y_pred - i * 1 * std_y_pred,
                             mean_y_pred - (i + 1) * 1 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='orchid')  # wheat
            plt.fill_between(original_x_test.squeeze().cpu(), mean_y_pred + i * 1 * std_y_pred,
                             mean_y_pred + (i + 1) * 1 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='orchid')
        plt.plot(original_x_train.cpu(), original_y_train.cpu(), 'ok', zorder=10, ms=6, label='Observations')
        # plt.grid(True)
        # plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
        #                 labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        ############sin#########
        # plt.axvline(-2, linestyle='--', color='darkorchid')   # darkorange
        # plt.axvline(2, linestyle='--', color='darkorchid')
        # plt.text(-1.5, -3, 'Training data range')
        #######gap
        # plt.axvline(-7.5, linestyle='--', color='darkorchid')
        # plt.axvline(-2.5, linestyle='--', color='darkorchid')
        # plt.axvline(5, linestyle='--', color='darkorchid')
        # plt.axvline(7.5, linestyle='--', color='darkorchid')
        # plt.text(-7.4, -1, 'Training data range', fontsize='small')
        # plt.text(5.1, -1, 'Training data', fontsize='xx-small')
        # plt.text(5.1, -1.8, 'range', fontsize='xx-small')

        ######g3######
        plt.axvline(-0.75, linestyle='--', color='darkorchid')
        plt.axvline(-0.25, linestyle='--', color='darkorchid')
        plt.axvline(0.25, linestyle='--', color='darkorchid')
        plt.axvline(0.75, linestyle='--', color='darkorchid')
        plt.text(-0.75, -1.5, 'Training data range', fontsize='small')
        plt.text(0.25, 1, 'Training data range', fontsize='small')


        plt.legend(loc='upper center', ncol=3, fontsize='small')
        plt.title('fsgld posterior')
        plt.tight_layout()
        plt.ylim([dataset.y_min, dataset.y_max])
        plt.tight_layout()
        print("--------------------------------------------------------------------------------------------")
        plt.savefig(figures_folder + '/plot_epoch{}.pdf'.format(epoch))
        print("figure saved")
        print("epoch : {} \t\t test loss \t\t : {}".format(epoch, test_loss))
        print("--------------------------------------------------------------------------------------------")

    ##############
    if epoch > 2000 and (epoch%100)==0:    # 3000_100, 2000_100
        print('save!')
        bnn.cpu()
        # torch.save(bnn.state_dict(), save_path + '/g3_rbf_%i.pt'%(mt))
        dd = f'{mt}'
        print(dd)
        save_path = results_folder + '/' + bnn_name_string + '/' + 'fsgld_toy_samples' + '/' + 'g3_' + dd      # sgld
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        bnn.save(save_path)
        mt += 1
        bnn.to(device)

########## calibrate curve ###############
# oy = original_y_test
# oy = oy.cpu().numpy()
# # upper_bound = mean_y_pred + 1.96 * std_y_pred / test_num
# # lower_bound = mean_y_pred - 1.96 * std_y_pred / test_num
# upper_bound = mean_y_pred + 5 * std_y_pred
# lower_bound = mean_y_pred - 5 * std_y_pred
# perc_within = np.mean((oy <= upper_bound)&(oy>=lower_bound))
# print(f'{perc_within*100:.1f}% of the points in the 95% confidence intervel')
#

log_f.close()

###########################
# plt.figure()
# indices = np.arange(10001)[::100]
# train_loss_all = np.array(train_loss_all)
# plt.plot(indices, train_loss_all[indices], '-ko', ms=3)
# plt.ylabel(r'log posterior')
# plt.tight_layout()
# plt.xlabel('Iteration')
# plt.tight_layout()
# plt.savefig(figures_folder + '/train_loss.pdf')

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")

def test():
    print('test')
