/**
 * @brief An implementation of the policyValueNet in PyTorch
 * @author Aurora
 * @date 2024-8-3
 */
#pragma once
#ifndef policyValueNet_
#define policyValueNet_

#include <tuple>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <chess.h>
#include <torch/torch.h>

namespace Network {
    /**
     * @brief Sets the learning rate to the given value
     */
    void setLearningRate(torch::optim::Adam* optimizer, double learningRate);

    /**
     * @brief policy-value network module
     * @class Net
     */
    class Net : public torch::nn::Module {
    public:
        Net();

        std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& stateInput);

    private:
        // common layers
        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::Conv2d conv3;

        // action policy layers
        torch::nn::Conv2d actConv1;
        torch::nn::Linear actFc1;

        // state value layers
        torch::nn::Conv2d valConv1;
        torch::nn::Linear valFc1;
        torch::nn::Linear valFc2;
    };

    /**
     * @brief policy-value network
     * @class PolicyValueNet
     */
     class PolicyValueNet {
     public:
         PolicyValueNet();

         /**
          * @param state_batch - a batch of states
          * @return a batch of action probabilities and state values
          */
         std::tuple<torch::Tensor, torch::Tensor> policy_value(std::vector<std::vector<float>> state_batch);

         std::pair<std::map<int, double>, double> PolicyValueFn(Chessboard* state);

         std::tuple<torch::Scalar, torch::Scalar>  train_step(std::vector<std::vector<float>> state_batch,
                         std::vector<std::vector<float>> mcts_probs,
                         std::vector<float> winner_batch,
                         float lr);

         void saveModel(const std::string& modelFile);

     private:
         const float l2 = 1e-4;
         std::shared_ptr<Net> policyValueNet;
         torch::optim::Adam optimizer;
     };
}

#endif