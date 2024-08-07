#include "policyValueNet.h"
#include <torch/torch.h>
#include <tuple>
#include <vector>

using namespace std;
using namespace torch;
namespace Network {
    void setLearningRate(torch::optim::Adam* optimizer, double learningRate) {
        for (auto param_group : optimizer->param_groups()) {
            param_group.options().set_lr(learningRate);
        }
    }

    Net::Net() : conv1(nn::Conv2dOptions(4, 32, 3).padding(1)),
                 conv2(nn::Conv2dOptions(32, 64, 3).padding(1)),
                 conv3(nn::Conv2dOptions(64, 128, 3).padding(1)),
                 actConv1(nn::Conv2dOptions(128, 4, 1)),
                 actFc1(nn::LinearOptions(4 * 19 * 19, 19 * 19)),
                 valConv1(nn::Conv2dOptions(128, 2, 1)),
                 valFc1(nn::LinearOptions(2 * 19 * 19, 64)),
                 valFc2(nn::LinearOptions(64, 1)) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("actConv1", actConv1);
        register_module("valConv1", valConv1);
        register_module("actFc1", actFc1);
        register_module("valFc1", valFc1);
        register_module("valFc2", valFc2);
    }

    tuple<Tensor, Tensor> Net::forward(const Tensor& stateInput) {
        // common layers
        auto x = conv1->forward(stateInput).relu();
        x = conv2->forward(x).relu();
        x = conv3->forward(x).relu();

        // action policy layers
        auto x_act = actConv1->forward(x).relu();
        x_act = x_act.view({-1, 4 * 19 * 19});
        x_act = actFc1->forward(x_act).log_softmax(1);

        // state value layers
        auto x_val = valConv1->forward(x).relu();
        x_val = x_val.view({-1, 2 * 19 * 19});
        x_val = valFc1->forward(x_val).relu();
        x_val = valFc2->forward(x_val).tanh();

        return make_tuple(x_act, x_val);
    }

    PolicyValueNet::PolicyValueNet() :
        policyValueNet(make_shared<Net>(Net())),
        optimizer(policyValueNet->parameters(), optim::AdamOptions().weight_decay(l2)) {}

    std::tuple<torch::Tensor, torch::Tensor> PolicyValueNet::policy_value(std::vector<std::vector<float>> state_batch) {
        auto state_tensor = torch::from_blob(&state_batch[0][0], {static_cast<long>(state_batch.size()), static_cast<long>(state_batch[0].size()), 19, 19}, kFloat);
        auto [log_act_probs, value] = policyValueNet->forward(state_tensor);
        return make_tuple(log_act_probs.exp(), value);
    }

    pair<map<int, double>, double> PolicyValueNet::PolicyValueFn(Chessboard* state) {
        auto availablePos = state->availables();
        auto currentState = state->getState();
        currentState.contiguous();
        currentState.view({1, 4, 19, 19});
        auto currentState_ = autograd::make_variable(currentState, false);
        auto result = policyValueNet->forward({currentState_}); // tuple<Tensor, Tensor>
        auto logActProbs = get<0>(result);
        auto value = get<1>(result);

        auto actProbs = logActProbs.exp().view({-1});
        actProbs = actProbs / actProbs.sum();

        double value_ = value.item().toDouble();
        map<int, double> actProbsMap;
        vector<int> indices;
        indices.reserve(availablePos.size());
        for (const auto& pos : availablePos) {
            indices.push_back(pos.absolutePos());
        }
        auto t = TensorOptions().dtype(kInt);
        Tensor indicesTensor = torch::from_blob(indices.data(),
                                                {static_cast<long>(indices.size())},
                                                kInt);
        auto availableActProbs = index_select(actProbs, 0, indicesTensor);
        for (int i = 0; i < availablePos.size(); i++) {
            actProbsMap[availablePos[i].absolutePos()] = availableActProbs[i].item<double>();
        }

        return make_pair(actProbsMap, value_);
    }

    std::tuple<Scalar, Scalar> PolicyValueNet::train_step(  std::vector<std::vector<float>> state_batch,
                                                            std::vector<std::vector<float>> mcts_probs,
                                                            std::vector<float> winner_batch, float lr) {
        auto    stateBatch = torch::from_blob(
                    &state_batch[0][0],
                    {static_cast<long>(state_batch.size()), static_cast<long>(state_batch[0].size()), 19, 19},
                    kFloat32
                ),
                mctsProbs = torch::from_blob(
                    &mcts_probs[0][0],
                    {static_cast<long>(mcts_probs.size()), static_cast<long>(mcts_probs[0].size()), 19, 19},
                    kFloat32
                ),
                winnerBatch = torch::from_blob(
                    &winner_batch[0],
                    {static_cast<long>(winner_batch.size())},
                    kFloat32
                );

        optimizer.zero_grad();
        setLearningRate(&optimizer, lr);

        auto [logActProbs, value] = policyValueNet->forward(stateBatch);
        auto    valueLoss = mse_loss(value.view(-1), winnerBatch),
                policyLoss= -mean(torch::sum(mctsProbs * logActProbs, 1)),
                loss = valueLoss + policyLoss;

        loss.backward();
        optimizer.step();

        auto entropy = -mean(torch::sum(torch::exp(logActProbs) * logActProbs), 1);

        return make_tuple(loss.item(), entropy.item());
    }

    void PolicyValueNet::saveModel(const std::string& modelFile) {
        serialize::OutputArchive archive;
        policyValueNet->save(archive);
        archive.save_to(modelFile);
    }
}