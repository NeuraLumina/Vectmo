#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>
#include <optional>

namespace VectmoErrors {
    const std::string FILE_NOT_CREATED_ERROR    = "FILE_NOT_CREATED_ERROR";
    const std::string ERROR_ON_WRITING_TO_FILE  = "ERROR_ON_WRITING_TO_FILE";
    const std::string ERROR_FILENAME_REQUIRED   = "ERROR_FILENAME_REQUIRED";
}

// Precompute character indices for O(1) lookup
class CharIndexMap {
private:
    static constexpr std::array<char, 96> SUPPORTED_ASCII_CHARS = {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ', '\n'};

    std::array<int, 256> lookupTable;

public:
    static constexpr int VOCAB_SIZE = 96;
    static constexpr int INVALID_INDEX = -1;

    CharIndexMap() {
        lookupTable.fill(INVALID_INDEX);
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            lookupTable[static_cast<unsigned char>(SUPPORTED_ASCII_CHARS[i])] = i;
        }
    }

    int operator()(char c) const {
        return lookupTable[static_cast<unsigned char>(c)];
    }

    char operator[](int index) const {
        return (index >= 0 && index < VOCAB_SIZE) ? SUPPORTED_ASCII_CHARS[index] : '\0';
    }

    bool isSupported(char c) const {
        return operator()(c) != INVALID_INDEX;
    }
};

// Fixed-size vector for character histogram
class CharHistogram {
private:
    std::array<double, CharIndexMap::VOCAB_SIZE> data{};

public:
    CharHistogram() = default;

    explicit CharHistogram(const std::string& word, const CharIndexMap& charMap) {
        for (char c : word) {
            int idx = charMap(c);
            if (idx != CharIndexMap::INVALID_INDEX) {
                data[idx] += 1.0;
            }
        }
    }

    double dot(const CharHistogram& other) const {
        return std::inner_product(data.begin(), data.end(), other.data.begin(), 0.0);
    }

    double magnitude() const {
        double sum = std::accumulate(data.begin(), data.end(), 0.0, 
            [](double acc, double val) { return acc + val * val; });
        return std::sqrt(sum);
    }

    double cosineSimilarity(const CharHistogram& other) const {
        double magA = magnitude();
        double magB = other.magnitude();
        
        if (magA == 0.0 || magB == 0.0) return 0.0;
        
        return dot(other) / (magA * magB);
    }

    const auto& getData() const { return data; }
};

// Core model data
class VectmoModel {
private:
    std::map<char, std::map<char, int>> bigramTable;
    std::set<std::string> vocabulary;
    std::map<std::string, CharHistogram> cachedEmbeddings;
    CharIndexMap charMap;

public:
    void train(const std::string& text) {
        buildBigramTable(text);
        buildVocabulary(text);
        cacheEmbeddings();
    }

    void buildBigramTable(const std::string& text) {
        bigramTable.clear();
        for (size_t i = 0; i + 1 < text.size(); ++i) {
            if (charMap.isSupported(text[i]) && charMap.isSupported(text[i + 1])) {
                bigramTable[text[i]][text[i + 1]]++;
            }
        }
    }

    void buildVocabulary(const std::string& text) {
        vocabulary.clear();
        std::istringstream stream(text);
        std::string token;
        while (stream >> token) {
            if (!token.empty()) {
                vocabulary.insert(token);
            }
        }
    }

    void cacheEmbeddings() {
        cachedEmbeddings.clear();
        for (const auto& word : vocabulary) {
            cachedEmbeddings[word] = CharHistogram(word, charMap);
        }
    }

    bool hasBigram(char c) const {
        return bigramTable.find(c) != bigramTable.end() && !bigramTable.at(c).empty();
    }

    std::vector<char> getTopFollowers(char c, int maxCount = 0) const {
        auto it = bigramTable.find(c);
        if (it == bigramTable.end()) return {};

        std::vector<std::pair<char, int>> followers(it->second.begin(), it->second.end());
        std::sort(followers.begin(), followers.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<char> result;
        for (const auto& [ch, count] : followers) {
            result.push_back(ch);
            if (maxCount > 0 && result.size() >= static_cast<size_t>(maxCount)) break;
        }
        return result;
    }

    std::optional<std::string> findMostSimilarWord(const std::string& word) const {
        if (vocabulary.empty()) return std::nullopt;

        CharHistogram wordHist(word, charMap);
        
        auto bestIt = vocabulary.begin();
        double bestScore = -1.0;
        
        for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
            auto cachedIt = cachedEmbeddings.find(*it);
            if (cachedIt == cachedEmbeddings.end()) continue;
            
            double score = wordHist.cosineSimilarity(cachedIt->second);
            
            if (score > bestScore ||
                (score == bestScore && 
                 std::abs(static_cast<int>(it->size()) - static_cast<int>(word.size())) <
                 std::abs(static_cast<int>(bestIt->size()) - static_cast<int>(word.size())))) {
                bestScore = score;
                bestIt = it;
            }
        }
        
        return *bestIt;
    }

    bool save(const std::string& basePath) const {
        // Save bigram table
        std::ofstream bigramFile(basePath + ".txt");
        if (!bigramFile) return false;
        
        for (const auto& [from, followers] : bigramTable) {
            int fromIdx = charMap(from);
            if (fromIdx == CharIndexMap::INVALID_INDEX) continue;
            
            for (const auto& [to, count] : followers) {
                int toIdx = charMap(to);
                if (toIdx == CharIndexMap::INVALID_INDEX) continue;
                bigramFile << fromIdx << ' ' << toIdx << ' ' << count << '\n';
            }
        }
        
        // Save vocabulary
        std::ofstream vocabFile(basePath + ".words");
        if (!vocabFile) return false;
        
        for (const auto& word : vocabulary) {
            vocabFile << word << '\n';
        }
        
        return true;
    }

    bool load(const std::string& basePath) {
        // Load bigram table
        std::ifstream bigramFile(basePath + ".txt");
        if (!bigramFile) return false;
        
        bigramTable.clear();
        int fromIdx, toIdx, count;
        while (bigramFile >> fromIdx >> toIdx >> count) {
            char from = charMap[fromIdx];
            char to = charMap[toIdx];
            if (from != '\0' && to != '\0') {
                bigramTable[from][to] = count;
            }
        }
        
        // Load vocabulary
        std::ifstream vocabFile(basePath + ".words");
        if (!vocabFile) return false;
        
        vocabulary.clear();
        std::string line;
        while (std::getline(vocabFile, line)) {
            if (!line.empty()) {
                vocabulary.insert(line);
            }
        }
        
        // Rebuild cache
        cacheEmbeddings();
        
        return true;
    }

    bool isTrained() const {
        return !bigramTable.empty() && !vocabulary.empty();
    }

    const auto& getVocabulary() const { return vocabulary; }
};

// Prediction engine
class VectmoPredictor {
private:
    const VectmoModel& model;
    static constexpr int CYCLE_WINDOW_SIZE = 6;

    static bool wouldCreateCycle(const std::string& sequence, char candidate) {
        if (static_cast<int>(sequence.size()) < CYCLE_WINDOW_SIZE) return false;
        
        std::string newSeq = sequence + candidate;
        std::string window = newSeq.substr(newSeq.size() - CYCLE_WINDOW_SIZE);
        size_t earlierPos = newSeq.find(window);
        size_t lastPos = newSeq.size() - CYCLE_WINDOW_SIZE;
        
        return earlierPos < lastPos;
    }

public:
    explicit VectmoPredictor(const VectmoModel& m) : model(m) {}

    std::string generateRawSequence(char seed, int maxChars) const {
        std::string result(1, seed);
        char current = seed;

        for (int i = 0; i < maxChars; ++i) {
            if (!model.hasBigram(current)) break;

            auto followers = model.getTopFollowers(current);
            char chosen = '\0';

            for (char candidate : followers) {
                if (!wouldCreateCycle(result, candidate)) {
                    chosen = candidate;
                    break;
                }
            }

            if (chosen == '\0' && !followers.empty()) {
                chosen = followers[0];  // forced move
            }

            if (chosen == '\0') break;

            result += chosen;
            current = chosen;
        }

        return result;
    }

    std::string snapToVocabulary(const std::string& rawSequence) const {
        if (rawSequence.size() <= 1) return rawSequence;

        std::vector<std::string> tokens;
        std::string current;
        
        for (char c : rawSequence) {
            if (c == ' ') {
                tokens.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        tokens.push_back(current);

        std::string result;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) result += ' ';
            
            if (tokens[i].empty()) continue;

            auto bestWord = model.findMostSimilarWord(tokens[i]);
            if (bestWord) {
                result += *bestWord;
            } else {
                result += tokens[i];  // fallback
            }
        }

        return result;
    }
};

// Main API class (clean interface)
class Vectmo {
private:
    VectmoModel model;
    std::string workingFileBase;
    bool modelLoaded = false;

public:
    bool setWorkingFile(const std::string& fileName) {
        if (fileName.empty()) {
            std::cerr << "[ERROR] " << VectmoErrors::ERROR_FILENAME_REQUIRED << '\n';
            return false;
        }
        workingFileBase = fileName;
        return true;
    }

    bool createFile() {
        if (workingFileBase.empty()) {
            std::cerr << "[ERROR] " << VectmoErrors::ERROR_FILENAME_REQUIRED << '\n';
            return false;
        }

        std::ofstream file(workingFileBase + ".txt", std::ios::trunc);
        if (!file) {
            std::cerr << "[ERROR] " << VectmoErrors::FILE_NOT_CREATED_ERROR << '\n';
            return false;
        }
        return true;
    }

    bool pretrainModel(const std::string& trainingText) {
        if (workingFileBase.empty()) {
            std::cerr << "[PRETRAIN] ERROR: No file set. Call setWorkingFile() first.\n";
            return false;
        }

        model.train(trainingText);
        
        if (!model.save(workingFileBase)) {
            std::cerr << "[PRETRAIN] ERROR: Failed to save model\n";
            return false;
        }

        modelLoaded = true;
        std::cout << "[PRETRAIN] Model trained and saved successfully\n";
        return true;
    }

    std::string predictNextText(const std::string& inputText, int maxChars = 50) {
        if (inputText.empty()) return "[No input provided]";

        if (!modelLoaded) {
            if (!model.load(workingFileBase)) {
                return "[Model not trained yet or file not found]";
            }
            modelLoaded = true;
        }

        char seed = inputText.back();
        
        VectmoPredictor predictor(model);
        std::string rawSequence = predictor.generateRawSequence(seed, maxChars);
        
        if (rawSequence.size() <= 1) return "[No continuation found]";
        
        std::string rawOutput = rawSequence.substr(1);  // remove seed
        std::string snapped = predictor.snapToVocabulary(rawOutput);
        
        return snapped;
    }
};

// Clean UI separation
class VectmoUI {
private:
    Vectmo vectmo;

    void printHeader() const {
        std::cout << "\n==== Vectmo: Text Vectorization & Prediction ====\n\n";
    }

    void printPredictionBox(const std::string& input, const std::string& output) const {
        std::cout << "\n┌─ PREDICTION ─────────────────────────┐\n";
        std::cout << "│ Input:  \"" << input << "\"\n";
        std::cout << "│ Output: \"" << output << "\"\n";
        std::cout << "└───────────────────────────────────────┘\n";
    }

public:
    void run() {
        printHeader();

        // Setup
        std::cout << "Enter filename base (default: vectmo_training_data): ";
        std::string filename;
        std::getline(std::cin, filename);
        
        if (filename.empty()) filename = "vectmo_training_data";
        
        if (!vectmo.setWorkingFile(filename) || !vectmo.createFile()) {
            return;
        }

        // Training
        std::cout << "Enter text to train the model: ";
        std::string trainText;
        std::getline(std::cin, trainText);

        if (trainText.empty()) {
            std::cerr << "[ERROR] No training text provided.\n";
            return;
        }

        if (!vectmo.pretrainModel(trainText)) {
            return;
        }

        // Prediction loop
        bool running = true;
        while (running) {
            std::cout << "\n╔════════════════════════════════════════╗\n";
            std::cout << "║          Prediction Mode                ║\n";
            std::cout << "╚════════════════════════════════════════╝\n";
            std::cout << "Enter starter text (or 'quit' to exit): ";
            
            std::string input;
            std::getline(std::cin, input);

            if (input == "quit" || input == "exit") {
                std::cout << "\nExiting. Status Code: 0\n";
                break;
            }

            if (input.empty()) {
                std::cout << "[WARNING] Empty input. Try again.\n";
                continue;
            }

            std::string prediction = vectmo.predictNextText(input);
            printPredictionBox(input, prediction);

            std::cout << "\nContinue? (Y/y/N/n): ";
            char choice;
            std::cin >> choice;
            std::cin.ignore();

            if (choice == 'N' || choice == 'n') {
                std::cout << "\nExiting. Status Code: 0\n";
                running = false;
            }
        }
    }
};

int main() {
    VectmoUI ui;
    ui.run();
    return 0;
}
