#include "MCTS.h"
#include <utility>
#include <random>
#include <algorithm>
#include <numeric>
#include <torch/torch.h>

using namespace std;
using namespace torch;

vector<double> softmax(const vector<double>& x) {
    double maxVal = *max_element(x.begin(), x.end());
    vector<double> expX(x.size());
    double sumExp = 0.0;

    for (int i = 0; i < x.size(); i++) {
        expX[i] = exp(x[i] - maxVal);
        sumExp += expX[i];
    }
    for (double& val : expX) {
        val /= sumExp;
    }

    return expX;
}
static Complex toComplex(int absolutePos) {
    Complex result;
    int realPart = absolutePos / 19 + 1;
    int imaginaryPart = absolutePos % 19 + 1;
    result.realPart = realPart;
    result.imaginaryPart = imaginaryPart;
    return result;
}
vector<double> dirichletDistribution(const vector<double>& alpha) {
    random_device rd;
    std::mt19937 gen(rd());
    vector<double> samples(alpha.size());

    for (size_t i = 0; i < alpha.size(); ++i) {
        gamma_distribution<double> dist(alpha[i], 1.0); // Instantiate gamma distribution
        samples[i] = dist(gen); // Generate a sample from this distribution
    }

    double sum = accumulate(samples.begin(), samples.end(), 0.0);
    for (double& sample : samples) {
        sample /= sum; // Normalize to ensure it's a probability distribution
    }

    return samples;
}
int weightedRandomChoice(const std::vector<int>& acts, const std::vector<double>& probs) {
    random_device rd;
    std::mt19937 gen(rd());
    discrete_distribution<> dist(probs.begin(), probs.end());

    return acts[dist(gen)];
}

namespace AlphaGo {
    TreeNode::TreeNode(double priorProbability, TreeNode *parent) : parent(parent), P(priorProbability) {}

    void TreeNode::expand(const std::map<int, double>& actionPriors) {
        for (const auto& actionPrior : actionPriors) {
            int action = actionPrior.first;
            double prob = actionPrior.second;
            if (children.find(action) == children.end()) {
                children.insert(pair<int, TreeNode>(action, TreeNode(prob, this)));
            }
        }
    }

    pair<int, TreeNode> & TreeNode::select(double c_puct) {
        vector<pair<int, TreeNode>> items;
        for (auto& item : children) {
            items.emplace_back(item);
        }

        sort(items.begin(), items.end(),
             [&](pair<int, TreeNode>& a, pair<int, TreeNode>& b) {
            double b_value = b.second.getValue(c_puct);
            double a_value = a.second.getValue(c_puct);
            if (b_value == a_value) {
                return a.first < b.first;
            }
            return a_value > b_value;
        });

        return items[0];
    }

    void TreeNode::update(double leafValue) {
        // count visit
        n_visits++;
        //  update Q, a running average of values for all visits.
        Q += 1.0*(leafValue - Q) / n_visits;
    }

    void TreeNode::updateRecursive(double leafValue) {
        // If it is not root, this node's parent should be updated first.
        if (parent != nullptr) {
            parent->updateRecursive(leafValue);
        }
        update(leafValue);
    }

    double TreeNode::getValue(double c_puct) {
        u = (c_puct * P * sqrt(parent->n_visits) / (1 + n_visits));
        return Q + u;
    }

    bool TreeNode::isLeaf() const {
        return children.empty();
    }

    bool TreeNode::isRoot() const {
        return (parent == nullptr);
    }

    MCTS::MCTS(std::pair<std::map<int, double>, double> (*policyValueFn)(Chessboard *), double c_puct, int n_playout) :
            root(TreeNode(1.0, nullptr)),
            policy(policyValueFn),
            c_puct(c_puct),
            n_playout(n_playout) {}

    void MCTS::playout(Chessboard *state) {
        auto node = root;
        while (true) {
            if (node.isLeaf()) {
                break;
            }
            auto action = node.select(c_puct).first;
            node = node.select(c_puct).second;
            state->drop(state->whoTurn, toComplex(action));
        }
        auto pair = policy(state); auto action_probs = pair.first; auto leaf_value = pair.second;
        bool end = state->sumChess > 300;
        int winner = end ? state->whoWin() : 0;

        if (!end) {
            node.expand(action_probs);
        } else {
            if (winner == 0) {
                leaf_value = 0.0;
            } else {
                leaf_value = (
                        (winner == state->whoTurn ? 1.0 : -1.0)
                );
            }
        }
        node.updateRecursive(-leaf_value);
    }

    pair<vector<int>, vector<double>> MCTS::getMoveProbs(Chessboard* state, double temp) {
        for (int n = 0; n < n_playout; n++) {
            auto stateCopy = state;
            playout(stateCopy);
        }

        vector<pair<int, double>> actVisits;
        for (auto& pair : root.children) {
            actVisits.emplace_back(pair.first, pair.second.n_visits);
        }

        vector<int> acts; vector<double> visits;
        for (auto& p : actVisits) {
            acts.push_back(p.first);
            visits.push_back(p.second);
        }
        torch::Tensor visitsTensor = torch::from_blob(visits.data(), {static_cast<long>(visits.size())}, torch::kDouble);
        visitsTensor = visitsTensor + 1e-10;
        visitsTensor = visitsTensor.log();
        visitsTensor = visitsTensor / temp;
        vector<double> actProbs(visitsTensor.data_ptr<double>(),
                                visitsTensor.data_ptr<double>() + visitsTensor.numel());
        actProbs = softmax(actProbs);

        return pair{acts, actProbs};
    }

    void MCTS::updateWithMove(int lastMove) {
        if (lastMove != -1) {
            if (root.children.find(lastMove) != root.children.end()) {
                root = root.children.at(lastMove);
                root.children.clear();
            } else {
                root = TreeNode(1.0f, nullptr);
            }
        } else {
            root = TreeNode(1.0f, nullptr);
        }
    }

    MCTSPlayer::MCTSPlayer(std::pair<std::map<int, double>, double> (*policyValueNet)(Chessboard *), double c_puct,
                           int n_playout, bool isSelfPlay) :
                    mcts(MCTS(policyValueNet, c_puct, n_playout)), isSelfPlay(isSelfPlay) {}

    void MCTSPlayer::setPlayerInd(int playerInd) {
        player = playerInd;
    }

    void MCTSPlayer::resetPlayer() {
        mcts.updateWithMove(-1);
    }

    std::pair<int, torch::Tensor> MCTSPlayer::getAction(Chessboard *board, double temp, bool returnProb) {
        auto sensibleMoves = board->availables();
        Tensor moveProbs = torch::zeros({19 * 19}, kDouble);

        if (!sensibleMoves.empty()) {
            auto [acts, probs] = mcts.getMoveProbs(board, temp);

            auto actsTensor = torch::from_blob(acts.data(), {static_cast<long>(acts.size())}, kInt64),
                 probsTensor = torch::from_blob(probs.data(), {static_cast<long>(probs.size())}, kDouble);
            moveProbs.index_put_({actsTensor}, probsTensor);
            int move;
            if (isSelfPlay) {
                double noiseAlpha = 0.3f, epsilon = 0.25f;
                vector<double> noise = dirichletDistribution(vector<double>(probs.size(), noiseAlpha));
                vector<double> noisyProbs(probs.size());
                transform(probs.begin(), probs.end(), noise.begin(), noisyProbs.begin(),
                          [epsilon](double p, double n) { return (1 - epsilon) * p + epsilon * n; });
                noisyProbs = softmax(noisyProbs);
                move = weightedRandomChoice(acts, noisyProbs);
                mcts.updateWithMove(move);
            } else {
                move = weightedRandomChoice(acts, probs);
                mcts.updateWithMove(-1);
            }

            if (returnProb) {
                return {move, moveProbs};
            } else {
                return {move, torch::zeros({19 * 19}, kDouble)};
            }
        } else {
            // The Board is full
            return {};
        }
    }
}

namespace pure {

    TreeNode::TreeNode(TreeNode *parent, double priorProbability) :
        parent(parent), P(priorProbability) {}

    void TreeNode::expand(const std::map<int, double> actionPriors) {
        for (const auto& actionPrior : actionPriors) {
            int action = actionPrior.first;
            double prob = actionPrior.second;
            if (children.find(action) == children.end()) {
                children.insert(pair(action, TreeNode(this, prob)));
            }
        }
    }

    pair<int, TreeNode> TreeNode::select(double c_puct) {
        vector<pair<int, TreeNode>> items;
        for (auto& item : children) {
            items.emplace_back(item);
        }

        sort(items.begin(), items.end(),
             [&](pair<int, TreeNode>& a, pair<int, TreeNode>& b) {
                 double b_value = b.second.getValue(c_puct);
                 double a_value = a.second.getValue(c_puct);
                 if (b_value == a_value) {
                     return a.first < b.first;
                 }
                 return a_value > b_value;
             });

        return items[0];
    }

    void TreeNode::update(double leafValue) {
        // count visit
        n_visits++;
        //  update Q, a running average of values for all visits.
        Q += 1.0*(leafValue - Q) / n_visits;
    }

    void TreeNode::updateRecursive(double leafValue) {
        // If it is not root, this node's parent should be updated first.
        if (parent != nullptr) {
            parent->updateRecursive(leafValue);
        }
        update(leafValue);
    }

    double TreeNode::getValue(double c_puct) {
        u = (c_puct * P * sqrt(parent->n_visits) / (1 + n_visits));
        return Q + u;
    }

    bool TreeNode::isLeaf() const {
        return children.empty();
    }

    bool TreeNode::isRoot() const {
        return (parent == nullptr);
    }

    MCTS::MCTS(std::pair<std::map<int, double>, double> (*policyValueFn)(Chessboard *), double c_puct, int n_playout) :
        root(TreeNode(nullptr, 1.0f)), policy(policyValueFn), c_puct(c_puct), n_playout(n_playout) {}

    void MCTS::playout(Chessboard *state) {
        auto node = root;
        while (true) {
            if (node.isLeaf()) break;
            auto action = node.select(c_puct).first;
            node = node.select(c_puct).second;
            state->drop(state->whoTurn, toComplex(action));
        }
        auto actionProbs = policy(state).first;
        bool end = (state->sumChess > 300);
        int winner = end ? state->whoWin() : 0;
        if (!end) node.expand(actionProbs);
        int leafValue = evaluateRollout(state);
        node.updateRecursive(-leafValue);
    }
}
