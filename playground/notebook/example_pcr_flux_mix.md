---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: julia 1.7.2
    language: julia
    name: julia-1.7
---

```julia
using Random

using Flux
using Optimisers
using Flux.Data:DataLoader
using MLDatasets
using Images
using ProgressMeter
using MLDatasets
```

```julia
using PyCallChainRules.Torch: TorchModuleWrapper, torch
```

```julia
Base.@kwdef mutable struct Args
    η = 3e-4
    λ = 0
    batchsize=128
    epochs=5
    seed=12345
    use_cuda=false
end

args = Args()
Random.seed!(args.seed)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
```

```julia
xtrain, ytrain = MLDatasets.MNIST(split=:train, Tx=Float32)[:]
xtest, ytest = MLDatasets.MNIST(split=:test, Tx=Float32)[:];
```

```julia
xtrain = Flux.unsqueeze(xtrain, dims=3) # (28, 28, 60000) -> (28, 28, 1 , 60000)
xtest = Flux.unsqueeze(xtest, dims=3)   # (28, 28, 10000) -> (28, 28, 1, 10000)
ytrain = Flux.onehotbatch(ytrain, 0:9) # (60000,) -> (10, 60000)
ytest = Flux.onehotbatch(ytest, 0:9)   # (10000,) -> (10, 10000)
train_loader = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
test_loader = DataLoader((xtest, ytest),  batchsize=128);
```

```julia
struct LeNet
    cnn_layer
    mlp_layer
    nclasses
end

Flux.@functor LeNet (cnn_layer, mlp_layer) # cnn_layer と mlp_layer が学習パラメータであることを指定する

(net::LeNet)(x) = x |> net.cnn_layer |> Flux.flatten |> net.mlp_layer

function create_model(imsize::Tuple{Int,Int,Int}, nclasses::Int)
    W, H, inC = imsize
    out_conv_size = (W ÷ 4 - 3, H ÷ 4 - 3, 16)
    cnn_layer = Chain(
        Conv((5,5), 1 => 6),
        x->Flux.relu.(x),
        MaxPool((2,2)),
        TorchModuleWrapper(torch.nn.Conv2d(6, 16, (5,5))),
        x->Flux.relu.(x),
        MaxPool((2,2))
    )
    mlp_layer = Chain(
        TorchModuleWrapper(torch.nn.Linear(prod(out_conv_size), 120)),
        x->Flux.relu.(x),
        Dense(120 => 84),
        x->Flux.relu.(x),
        TorchModuleWrapper(torch.nn.Linear(84,nclasses)),
    )
    LeNet(cnn_layer, mlp_layer, nclasses)
end
```

```julia
model = create_model((28, 28, 1), 10) |> f32

opt = Optimisers.ADAM(0.01)
state = Optimisers.setup(opt, model)
loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)
```

```julia
for e in 1:args.epochs
    @showprogress for (x, y) in train_loader
        gs, _ = Flux.gradient(model, x, y) do m, x, y
            ŷ = m(x)
            loss(ŷ, y)
        end
        state, model = Optimisers.update(state, model, gs)
    end
    acc = sum(Flux.onecold(model(xtest)) .== Flux.onecold(ytest))
    acc /= size(ytest, 2)
    @info("acc", 100acc, "%")
end
```

```julia
x_test, y_test = first(test_loader);
img = x_test[:, :, :, 1]
label = Flux.onecold(y_test[:, 1]) - 1
println("pred:", Flux.onecold(model(x_test))[1]-1)
println("actual:", label)
Gray.(dropdims(img, dims=3) |> transpose)
```
