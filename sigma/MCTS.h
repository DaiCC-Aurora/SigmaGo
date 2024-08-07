/**
 * @brief Monte Carlo tree search
 * @author Aurora
 * @date 2024-8-4
 */
#pragma once
#ifndef MCTS_
#define MCTS_

#include <cmath>
#include <numeric>
#include <vector>
#include <map>
#include <set>
#include <chess.h>
#include <torch/torch.h>

/**
     * @brief 元素的指数与所有元素指数和的比值
     * @param x - 输入元素
     */
std::vector<double> softmax(const std::vector<double>& x);
/**
 * @brief 将绝对位置转为向量
 * @param absolutePos - 绝对位置
 * @return 向量
 * @note 假设棋盘大小为19 * 19
 */
static Complex toComplex(int absolutePos);
std::vector<double> dirichletDistribution(const std::vector<double>& alpha);
int weightedRandomChoice(const std::vector<int>& acts, const std::vector<double>& probs);

namespace AlphaGo {
    class TreeNode {
    public:
        /**
         * @brief 构造函数
         * @param priorProbability - 先验概率
         * @param parent - 父类
         */
        explicit TreeNode(double priorProbability, TreeNode *parent = nullptr);

        /**
         * @brief Expand tree by creating new children.
         * @brief 通过创建新的子节点来扩展树
         * @param actionPriors - 动作元组及其先验概率的列表  according to the policy function.
         */
        void expand(const std::map<int, double>& actionPriors);
        /**
         * @brief Select action among children that gives maximum action value Q plus bonus u(P).
         * @param c_puct - 剪枝概率
         * @return pair<int, TreeNode>
         */
        std::pair<int, TreeNode> & select(double c_puct);
        /**
         * @brief Update node values from leaf evaluation.
         * @param leafValue - the value of subtree evaluation from the current player's perspective.
         */
        void update(double leafValue);
        /**
         * @brief Like a call to update(), but applied recursively for all ancestors.
         */
        void updateRecursive(double leafValue);
        /**
         * @brief Calculate and return the value for this node. It is a combination of leaf evaluations Q,
         * and this node's prior adjusted for its visit count, u.
         * @param c_puct - a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
         * @return double
         */
        double getValue(double c_puct);
        /**
         * @brief Check if leaf node.
         * @note no nodes below this have been expanded
         */
        [[nodiscard]] bool isLeaf() const;
        [[nodiscard]] bool isRoot() const;

        TreeNode *parent;
        std::map<int, TreeNode> children;
        double n_visits = 0.0f;
        double Q = 0,
               u = 0,
               P;
    };

    /**
     * @brief An implementation of Monte Carlo Tree Search.
     * @class MCTS
     */
    class MCTS {
    public:
        explicit MCTS(std::pair<std::map<int, double>, double> (*policyValueFn)(Chessboard*),
                double c_puct = 5.0f, int n_playout = 10000);

        /**
         * @brief Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
         * @param state - the chessboard
         */
        void playout(Chessboard* state);
        std::pair<std::vector<int>, std::vector<double>> getMoveProbs(Chessboard* state, double temp = 1e-3);
        void updateWithMove(int lastMove);


    private:
        TreeNode root;
        std::pair<std::map<int, double>, double> (*policy)(Chessboard*);
        double c_puct;
        int n_playout;
    };

    class MCTSPlayer {
    public:
        MCTSPlayer( std::pair<std::map<int, double>, double> (*policyValueNet)(Chessboard*),
                    double c_puct = 5, int n_playout = 2000, bool isSelfPlay = false);

        void setPlayerInd(int playerInd);
        void resetPlayer();
        std::pair<int, torch::Tensor> getAction(Chessboard* board, double temp = 1e-3, bool returnProb = false);

    private:
        MCTS mcts;
        bool isSelfPlay;
        int player;
    };
}

namespace pure {
    std::pair<std::vector<Complex>, float> rolloutPolicyFn(Chessboard* chessboard);
    std::pair<std::pair<std::vector<Complex>, float>, int> policyValueFn(Chessboard* chessboard);

    class TreeNode {
    public:
        TreeNode(TreeNode* parent, double priorProbability);

        void expand(std::map<int, double> actionPriors);
        std::pair<int, TreeNode> select(double c_puct);
        void update(double leafValue);
        void updateRecursive(double leafValue);
        double getValue(double c_puct);
        [[nodiscard]] bool isLeaf() const;
        [[nodiscard]] bool isRoot() const;

    private:
        TreeNode* parent;
        std::map<int, TreeNode> children;
        int n_visits = 0;
        double  Q = 0,
                u = 0,
                P;
    };

    class MCTS {
    public:
        explicit MCTS(std::pair<std::map<int, double>, double>(*policyValueFn)(Chessboard*),
                double c_puct = 5.0f, int n_playout = 10000);

        void playout(Chessboard* state);
        int evaluateRollout(Chessboard* state, int limit = 1000);
        std::pair<int, TreeNode> getMove();
        void updateWithMove(int lastMove);

    private:
        TreeNode root;
        std::pair<std::map<int, double>, double> (*policy)(Chessboard*);
        double c_puct;
        int n_playout;
    };

    class MCTSPlayer {
    public:
        MCTSPlayer(double c_puct = 5, int n_playout = 2000);
        void setPlayerInd();
        void resetPlayer();
        int getAction(Chessboard* board);

    private:
        MCTS mcts;
        bool isSelfPlay;
        int player;
    };
}
#endif