import torch
import pyro
import pyro.distributions as dist
from torch.autograd import Variable


def weather():
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]


for _ in range(3):
    print(weather())


def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = [200] if cloudy == 'sunny' and temp > 80.0 else [50]
    ice_cream = pyro.sample('ice_cream', dist.normal,
                            Variable(torch.Tensor(expected_sales)),
                            Variable(torch.Tensor([10.0])))
    return cloudy, temp, ice_cream.data[0]


for _ in range(3):
    print(ice_cream_sales())
