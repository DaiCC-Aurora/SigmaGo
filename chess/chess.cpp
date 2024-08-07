/**
 * @file chess.cpp
 * @brief 棋盘入口
 * @author Aurora
 * @date 2024-7-20
 */
#include <iostream>
#include <vector>
#include <stack>
#include <torch/torch.h>
#include "chess.h"

Complex::Complex() : realPart(0), imaginaryPart(0) {}
Complex::Complex(int x, int y) : realPart(x), imaginaryPart(y) {}

int Complex::absolutePos() const {
    int result = this->realPart + 19 * imaginaryPart;
    return result;
}

Complex Complex::operator+(const Complex &complex) const {
    return {this->realPart + complex.realPart, this->imaginaryPart + complex.imaginaryPart};
}
Complex Complex::operator-(const Complex &complex) const {
    return {this->realPart - complex.realPart, this->imaginaryPart - complex.imaginaryPart};
}

Chess::Chess() : color(0), position(Complex(0, 0)) {}
Chess::Chess(const int color, const Complex position) : color(color), position(position) {}
Chess::~Chess() = default;

Chessboard::Chessboard() {
    // 创建一个全零矩阵
    int x = 0, y = 0;
    for (const auto& i : chessboard) {
        for (auto j : i) {
            chessboard[x][y] = Chess(0, Complex(x, y));
            x++;
        }
        x = 0;
        y++;
    }

    whoTurn = -1;
    sumChess = 0;
}

void Chessboard::drop(int color, Complex position) {
    if (whoTurn != color) throw std::exception();   // 非该棋手落子
    if (getColor(position) != 0) throw std::exception();    // 该位置已经落子
    chessboard[position.realPart][position.imaginaryPart] = Chess(color, position);
    whoTurn = -whoTurn;
    sumChess++;

    std::array<std::array<Chess, 19>, 19> c;
    c[position.realPart][position.imaginaryPart].color = whoTurn;
    allStates.push_back(c);

    update();
}

int Chessboard::getColor(Complex position) {
    return chessboard[position.realPart][position.imaginaryPart].color;
}

void Chessboard::update() {
    std::vector<std::vector<bool>> visited(19, std::vector<bool>(19, false));

    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            Complex pos(i, j);
            int color(getColor(pos));
            if (color != 0 && !visited[pos.realPart][pos.imaginaryPart]) {
                std::vector<Complex> group;
                if (!DFS(pos, color, visited, group)) {
                    for (const auto& deadPos : group) {
                        chessboard[deadPos.realPart][deadPos.imaginaryPart] =
                                Chess(0, Complex(deadPos.realPart, deadPos.imaginaryPart));
                    }
                }
            }
        }
    }
}

int Chessboard::whoWin() {
    std::vector<std::vector<bool>> visited(19, std::vector<bool>(19, false));
    float blackTerritory = 0,
        whiteTerritory = 0;
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            Complex pos(i, j);
            if (!visited[pos.realPart][pos.imaginaryPart] && chessboard[i][j].color == 0) {
                int territoryColor = DFSForTerritory(pos, visited);
                if (territoryColor == 1) blackTerritory += 1;
                else if (territoryColor == 2) whiteTerritory += 1;
            }
        }
    }

    float komi = 7.5;
    float adjustedBlackTerritory = blackTerritory - komi;
    float adjustedWhiteTerritory = whiteTerritory;

    if (adjustedBlackTerritory > adjustedWhiteTerritory) {
        return -1;
    } else if (adjustedBlackTerritory < adjustedWhiteTerritory) {
        return 1;
    } else {
        return 0;
    }
}

bool Chessboard::DFS(const Complex &position, int color, std::vector<std::vector<bool>> &visited,
                     std::vector<Complex> &group) {
    if (position.realPart < 0 ||
        position.realPart >= 19 ||
        position.imaginaryPart < 0 ||
        position.imaginaryPart >= 19 ||
        visited[position.realPart][position.imaginaryPart] ||
        chessboard[position.realPart][position.imaginaryPart].color != color) {
        return false;
    }

    visited[position.realPart][position.imaginaryPart] = true;
    group.push_back(position);

    // 检查四个方向是否有气
    for (auto dir : directions) {
        Complex neighborPos = position + dir;
        if (chessboard[neighborPos.realPart][neighborPos.imaginaryPart].color == 0 &&
            neighborPos.realPart >= 0 &&
            neighborPos.realPart < 19 &&
            neighborPos.imaginaryPart >= 0 &&
            neighborPos.imaginaryPart < 19) {
            return true;
        }
    }

    // 没有找到 继续递归
    for (auto dir : directions) {
        Complex neighborPos = position + dir;
        if (DFS(neighborPos, color, visited, group) &&
                neighborPos.realPart >= 0 &&
                neighborPos.realPart < 19 &&
                neighborPos.imaginaryPart >= 0 &&
                neighborPos.imaginaryPart < 19) {
            return true;
        }
    }

    return false;
}

int Chessboard::DFSForTerritory(const Complex &position, std::vector<std::vector<bool>> &visited) {
    std::stack<Complex> positionsToCheck;
    positionsToCheck.push(position);

    int territoryColor = 0;
    bool hasBorder = false;

    while (!positionsToCheck.empty()) {
        Complex current = positionsToCheck.top();
        positionsToCheck.pop();

        if (current.realPart < 0 ||
            current.realPart >= 19 ||
            current.imaginaryPart < 0 ||
            current.imaginaryPart >= 19 ||
            visited[current.realPart][current.imaginaryPart]) {
            continue;
        }

        visited[current.realPart][current.imaginaryPart] = true;

        int neighborColor = chessboard[current.realPart][current.imaginaryPart].color;
        if (neighborColor != 0) {
            hasBorder = true;
            territoryColor += neighborColor;
        }

        // 检查邻接位置，将未访问过的空位置压入栈中
        for (auto dir : directions) {
            Complex neighborPos = current + dir;
            if (chessboard[neighborPos.realPart][neighborPos.imaginaryPart].color == 0 &&
                neighborPos.realPart >= 0 &&
                neighborPos.realPart < 19 &&
                neighborPos.imaginaryPart >= 0 &&
                neighborPos.imaginaryPart < 19 &&
                chessboard[neighborPos.realPart][neighborPos.imaginaryPart].color == 0 &&
                !visited[neighborPos.realPart][neighborPos.imaginaryPart]) {
                positionsToCheck.push(neighborPos);
            }
        }
    }

    // 如果领地周围有多种颜色的棋子，返回0（即不属于任何一方）
    if (hasBorder && abs(territoryColor) > 1) {
        return 0;
    }

    return territoryColor;
}

std::vector<Complex> Chessboard::availables() {
    std::vector<Complex> result;
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            if (chessboard[i][j].color == 0) {
                result.emplace_back(Complex(i, j));
            }
        }
    }
    return result;
}

torch::Tensor Chessboard::getState() {
    torch::Tensor state = torch::zeros({4, 19, 19}, torch::dtype(torch::kFloat32));

    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            Complex pos(i, j);
            int color = getColor(pos);
            if (color == whoTurn) {
                state.index_put_({0, i, j}, 1.0);
            } else if (color == -whoTurn) {
                state.index_put_({1, i, j}, 1.0);
            }
        }
    }

    if (sumChess > 0) {
        auto lastMovePos = allStates.back();
        for (int i = 0; i < 19; i++) {
            for (int j = 0; j < 19; j++) {
                Complex position(i, j);
                if (getColor(position) == 1 || getColor(position) == -1) {
                    state.index_put_({2, i, j}, 1.0);
                }
            }
        }
    }

    state.index_put_({3, torch::indexing::Slice(), torch::indexing::Slice()}, (whoTurn == 1) ? 1.0 : -1.0);

    return state;
}