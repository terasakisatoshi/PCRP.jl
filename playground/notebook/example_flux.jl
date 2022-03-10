# -*- coding: utf-8 -*-
# +
using Random

using Flux
using Flux.Data:DataLoader
using MLDatasets
using Images
using ProgressMeter

# +
Base.@kwdef mutable struct Args
    η = 0.05
    λ = 0
    batchsize=128
    epochs=5
    seed=12345
    use_cuda=false
end

args = Args()
Random.seed!(args.seed)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
# -

using Flux
using Flux.Data: DataLoader
using MLDatasets
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)
xtrain = Flux.unsqueeze(xtrain, 3) # (28, 28, 60000) -> (28, 28, 1 , 60000)
xtest = Flux.unsqueeze(xtest, 3)   # (28, 28, 10000) -> (28, 28, 1, 10000)
ytrain = Flux.onehotbatch(ytrain, 0:9) # (60000,) -> (10, 60000)
ytest = Flux.onehotbatch(ytest, 0:9)   # (10000,) -> (10, 10000)
train_loader = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
test_loader = DataLoader((xtest, ytest),  batchsize=128);

struct LeNet
    cnn_layer
    mlp_layer
    nclasses
end

Flux.@functor LeNet (cnn_layer, mlp_layer) # cnn_layer と mlp_layer が学習パラメータであることを指定する

# https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl
# を一部改変
function create_model(imsize::Tuple{Int,Int,Int}, nclasses::Int)
    W, H, inC = imsize
    out_conv_size = (W ÷ 4 - 3, H ÷ 4 - 3, 16)
    cnn_layer = Chain(
        Conv((5, 5), inC => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2))
    )
    mlp_layer = Chain(
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses),
    )
    LeNet(cnn_layer, mlp_layer, nclasses)
end

(net::LeNet)(x) = x |> net.cnn_layer |> flatten |> net.mlp_layer


model = create_model((28, 28, 1), 10) |> f32
ps = Flux.params(model);
opt = Descent(args.η)
loss(ŷ, y) = Flux.Losses.logitcrossentropy(ŷ, y)

for e in 1:args.epochs
    @showprogress for (x, y) in train_loader
        gs = Flux.gradient(ps) do
            ŷ = model(x)
            loss(ŷ, y)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    acc = sum(Flux.onecold(model(xtest)) .== Flux.onecold(ytest))
    acc /= size(ytest, 2)
    @info("acc", 100acc, "%")
end


x_test, y_test = first(test_loader);
img = x_test[:, :, :, 1]
label = Flux.onecold(y_test[:, 1]) - 1
println("pred:", Flux.onecold(model(x_test))[1]-1)
println("actual:", label)
Gray.(dropdims(img, dims=3) |> transpose)
