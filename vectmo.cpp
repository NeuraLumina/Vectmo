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

class VectmoErrors
{
protected:
    std::string FILE_NOT_CREATED_ERROR    = "FILE_NOT_CREATED_ERROR";
    std::string ERROR_ON_WRITING_TO_FILE  = "ERROR_ON_WRITING_TO_FILE";
    std::string ERROR_FILENAME_REQUIRED   = "ERROR_FILENAME_REQUIRED";
};

class Vectmo : private VectmoErrors
{
private:
    static constexpr std::array<char, 96> SUPPORTED_ASCII_CHARS = {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
        'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ', '\n'};

    static constexpr int VOCAB_SIZE = 96;   // matches SUPPORTED_ASCII_CHARS.size()

    // bigramTable[A][B] = number of times B followed A in the training data.
    std::map<char, std::map<char, int>> bigramTable;

    // Deduplicated vocabulary from training.
    std::set<std::string> pretrainedWords;

    std::string fileNameInternal          = "";
    const std::string fileNameInternalExt = ".txt";

    // ---------------------------------------------------------------------------
    bool isFileNameSet()
    {
        return !fileNameInternal.empty();
    }

    // ---------------------------------------------------------------------------
    int getCharIndex(char character)
    {
        for (int i = 0; i < VOCAB_SIZE; i++)
        {
            if (SUPPORTED_ASCII_CHARS[i] == character)
                return i;
        }
        return -1;
    }

    // ---------------------------------------------------------------------------
    // Builds bigramTable from raw text.
    // ---------------------------------------------------------------------------
    void buildBigramTable(const std::string &text)
    {
        bigramTable.clear();
        for (size_t i = 0; i + 1 < text.size(); i++)
            bigramTable[text[i]][text[i + 1]]++;
    }

    // ---------------------------------------------------------------------------
    // Extracts whitespace-delimited tokens into pretrainedWords.
    // ---------------------------------------------------------------------------
    void buildWordList(const std::string &text)
    {
        pretrainedWords.clear();
        std::istringstream stream(text);
        std::string token;
        while (stream >> token)
        {
            if (!token.empty())
                pretrainedWords.insert(token);
        }
    }

    // ---------------------------------------------------------------------------
    void saveBigramTable()
    {
        std::ofstream file(fileNameInternal, std::ios::trunc);
        if (!file.is_open()) return;

        for (auto &[from, followers] : bigramTable)
        {
            int fromIdx = getCharIndex(from);
            if (fromIdx == -1) continue;
            for (auto &[to, count] : followers)
            {
                int toIdx = getCharIndex(to);
                if (toIdx == -1) continue;
                file << fromIdx << " " << toIdx << " " << count << "\n";
            }
        }
        file.close();
    }

    // ---------------------------------------------------------------------------
    bool loadBigramTable()
    {
        bigramTable.clear();
        std::ifstream file(fileNameInternal);
        if (!file.is_open()) return false;

        int fromIdx, toIdx, count;
        bool anyRead = false;

        while (file >> fromIdx >> toIdx >> count)
        {
            if (fromIdx < 0 || fromIdx >= VOCAB_SIZE ||
                toIdx  < 0 || toIdx  >= VOCAB_SIZE)
                continue;
            bigramTable[SUPPORTED_ASCII_CHARS[fromIdx]][SUPPORTED_ASCII_CHARS[toIdx]] = count;
            anyRead = true;
        }
        file.close();
        return anyRead;
    }

    // ---------------------------------------------------------------------------
    void saveWordList()
    {
        std::string path = fileNameInternal.substr(0, fileNameInternal.size() - 4) + ".words";
        std::ofstream file(path, std::ios::trunc);
        if (!file.is_open()) return;
        for (auto &word : pretrainedWords)
            file << word << "\n";
        file.close();
    }

    // ---------------------------------------------------------------------------
    bool loadWordList()
    {
        std::string path = fileNameInternal.substr(0, fileNameInternal.size() - 4) + ".words";
        std::ifstream file(path);
        if (!file.is_open()) return false;

        pretrainedWords.clear();
        std::string line;
        bool anyRead = false;
        while (std::getline(file, line))
        {
            if (!line.empty())
            {
                pretrainedWords.insert(line);
                anyRead = true;
            }
        }
        file.close();
        return anyRead;
    }

    // ---------------------------------------------------------------------------
    // CHARACTER FREQUENCY HISTOGRAM EMBEDDING.
    //
    // Produces a fixed-length vector of size VOCAB_SIZE (96).  Slot i holds the
    // number of times SUPPORTED_ASCII_CHARS[i] appears in the word.
    //
    // "cat"  -> [... 0, 0, 1, 0, ... 1, ... 0, 1, 0 ...]
    //                     ^a            ^c      ^t
    //
    // Every word maps to the same 96-dimensional space regardless of length.
    // Shared characters produce non-zero dot products; characters that appear in
    // one word but not the other contribute 0.  This makes cosine similarity
    // meaningful: two words with no characters in common are orthogonal (score 0).
    // ---------------------------------------------------------------------------
    static std::vector<double> embedWord(const std::string &word)
    {
        std::vector<double> histogram(VOCAB_SIZE, 0.0);
        for (char c : word)
        {
            for (int i = 0; i < VOCAB_SIZE; i++)
            {
                if (SUPPORTED_ASCII_CHARS[i] == c)
                {
                    histogram[i] += 1.0;
                    break;
                }
            }
            // Characters not in the supported set are silently skipped.
        }
        return histogram;
    }

    // ---------------------------------------------------------------------------
    // Cosine similarity between two fixed-length vectors.
    //
    // cosine(A, B) = (A · B) / (‖A‖ * ‖B‖)
    //
    // Both vectors are guaranteed to be the same length (VOCAB_SIZE) because
    // they come from embedWord.  Returns 0.0 if either has zero magnitude
    // (i.e. the word contained no supported characters at all).
    // ---------------------------------------------------------------------------
    static double cosineSimilarity(const std::vector<double> &a,
                                   const std::vector<double> &b)
    {
        double dot  = 0.0;
        double magA = 0.0;
        double magB = 0.0;

        for (int i = 0; i < VOCAB_SIZE; i++)
        {
            dot  += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }

        magA = std::sqrt(magA);
        magB = std::sqrt(magB);

        if (magA == 0.0 || magB == 0.0) return 0.0;

        return dot / (magA * magB);
    }

    // ---------------------------------------------------------------------------
    // Scans pretrainedWords, returns the one with the highest cosine similarity
    // to `word` using histogram embeddings.  On a tie, the shorter word wins
    // (closer length = more structurally similar); ties on length keep the first
    // alphabetically (deterministic).
    // ---------------------------------------------------------------------------
    std::pair<std::string, double> findMostSimilarWord(const std::string &word)
    {
        std::vector<double> wordVec = embedWord(word);

        std::string bestWord  = word;
        double      bestScore = -1.0;   // cosine range is [-1, 1]; -1 is a safe floor

        for (auto &candidate : pretrainedWords)
        {
            std::vector<double> candVec = embedWord(candidate);
            double score = cosineSimilarity(wordVec, candVec);

            // Strictly better score, OR same score but candidate is closer in
            // length to the generated word (tiebreaker).
            if (score > bestScore ||
                (score == bestScore &&
                 std::abs(static_cast<int>(candidate.size()) - static_cast<int>(word.size())) <
                 std::abs(static_cast<int>(bestWord.size())  - static_cast<int>(word.size()))))
            {
                bestScore = score;
                bestWord  = candidate;
            }
        }

        return {bestWord, bestScore};
    }

    // ---------------------------------------------------------------------------
    // Splits on single spaces, preserving runs so reconstruction is lossless.
    // ---------------------------------------------------------------------------
    static std::vector<std::string> splitOnSpaces(const std::string &text)
    {
        std::vector<std::string> tokens;
        std::string current;
        for (char c : text)
        {
            if (c == ' ')
            {
                tokens.push_back(current);
                current.clear();
            }
            else
            {
                current += c;
            }
        }
        tokens.push_back(current);
        return tokens;
    }

    // ---------------------------------------------------------------------------
    // Post-processing: split raw bigram output on spaces, snap each non-empty
    // token to the nearest pretrained word by cosine on histogram embeddings,
    // reconstruct with original spacing.
    // ---------------------------------------------------------------------------
    std::string snapWordsToVocabulary(const std::string &rawPrediction)
    {
        std::vector<std::string> tokens = splitOnSpaces(rawPrediction);
        std::string result;

        for (size_t i = 0; i < tokens.size(); i++)
        {
            if (i > 0) result += ' ';

            if (tokens[i].empty())
                continue;   // consecutive/leading/trailing space — already re-inserted

            auto [bestWord, bestScore] = findMostSimilarWord(tokens[i]);

            std::cout << "  [SIM] \"" << tokens[i] << "\" -> \""
                      << bestWord << "\" (cosine=" << bestScore << ")" << std::endl;

            result += bestWord;
        }

        return result;
    }

public:
    // ---------------------------------------------------------------------------
    void setWorkingFile(const std::string &fileName)
    {
        if (fileName.empty())
        {
            std::cout << "[ERROR] " << ERROR_FILENAME_REQUIRED << std::endl;
            return;
        }
        fileNameInternal = fileName + fileNameInternalExt;
    }

    // ---------------------------------------------------------------------------
    std::map<std::string, std::string> createFile()
    {
        if (!isFileNameSet())
            return {{"fileName", ERROR_FILENAME_REQUIRED}};

        std::ofstream file(fileNameInternal, std::ios::trunc);
        if (file.fail())
            return {{"fileName", FILE_NOT_CREATED_ERROR}};

        file.close();
        return {{"fileName", fileNameInternal}};
    }

    // ---------------------------------------------------------------------------
    // Public-facing embedding: returns the histogram vector for a string.
    // Used by pretrainModel to write the .vec companion file.
    // ---------------------------------------------------------------------------
    std::vector<double> embedInputToVector(const std::string &text)
    {
        return embedWord(text);
    }

    // ---------------------------------------------------------------------------
    // Builds and persists bigram table + word list.  Writes .vec companion.
    // ---------------------------------------------------------------------------
    void pretrainModel(const std::string &trainingText)
    {
        if (!isFileNameSet())
        {
            std::cout << "[PRETRAIN] ERROR: No file set. Call setWorkingFile() first." << std::endl;
            return;
        }

        buildBigramTable(trainingText);
        saveBigramTable();
        std::cout << "[PRETRAIN] Bigram table saved to " << fileNameInternal
                  << " (" << bigramTable.size() << " unique source characters)" << std::endl;

        buildWordList(trainingText);
        saveWordList();
        std::cout << "[PRETRAIN] Word list saved (" << pretrainedWords.size() << " unique words): ";
        for (auto &w : pretrainedWords) std::cout << "\"" << w << "\" ";
        std::cout << std::endl;

        // .vec companion — full-text histogram for inspection
        std::string vecFileName = fileNameInternal.substr(0, fileNameInternal.size() - 4) + ".vec";
        std::ofstream vecFile(vecFileName, std::ios::trunc);
        if (vecFile.is_open())
        {
            std::vector<double> embedding = embedInputToVector(trainingText);
            for (size_t i = 0; i < embedding.size(); i++)
            {
                if (i > 0) vecFile << " ";
                vecFile << embedding[i];
            }
            vecFile << '\n';
            vecFile.close();
            std::cout << "[PRETRAIN] Embedding written to " << vecFileName << std::endl;
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 1: bigram chain (single-char lookahead + cycle detection).
    // Phase 2: snap every generated word to vocabulary via cosine on histograms.
    // ---------------------------------------------------------------------------
    std::string predictNextText(const std::string &inputText, int maxChars = 50)
    {
        if (inputText.empty())
            return "[No input provided]";

        if (bigramTable.empty())
        {
            if (!loadBigramTable())
                return "[Model not trained yet]";
        }

        if (pretrainedWords.empty())
            loadWordList();

        char current = inputText.back();

        std::cout << "[PREDICT] Seed character: '" << current << "' (0x"
                  << std::hex << static_cast<int>(static_cast<unsigned char>(current))
                  << std::dec << ")" << std::endl;

        // --- Phase 1: bigram chain ---
        std::string prediction(1, current);   // seed included for cycle checks
        constexpr int windowSize = 6;

        for (int i = 0; i < maxChars; i++)
        {
            auto rowIt = bigramTable.find(current);
            if (rowIt == bigramTable.end() || rowIt->second.empty())
                break;

            std::vector<std::pair<char, int>> followers(rowIt->second.begin(),
                                                        rowIt->second.end());
            std::sort(followers.begin(), followers.end(),
                      [](const std::pair<char, int> &a, const std::pair<char, int> &b)
                      { return a.second > b.second; });

            char chosen = '\0';
            bool found  = false;

            for (auto &[candidate, count] : followers)
            {
                (void)count;

                std::string hyp = prediction + candidate;

                if (static_cast<int>(hyp.size()) >= windowSize)
                {
                    std::string newWindow = hyp.substr(hyp.size() - windowSize);
                    size_t earlierPos = hyp.find(newWindow);
                    size_t lastPos    = hyp.size() - windowSize;

                    if (earlierPos < lastPos)
                        continue;   // cycle — skip
                }

                chosen = candidate;
                found  = true;
                break;
            }

            if (!found)
                chosen = followers[0].first;   // all blocked — forced move

            prediction += chosen;
            current    = chosen;
        }

        if (prediction.size() <= 1)
            return "[No continuation found]";

        std::string rawOutput = prediction.substr(1);   // strip seed

        std::cout << "[PREDICT] Raw bigram output: \"" << rawOutput << "\"" << std::endl;

        // --- Phase 2: snap to vocabulary ---
        if (pretrainedWords.empty())
            return rawOutput;

        std::string snapped = snapWordsToVocabulary(rawOutput);

        return snapped;
    }
};

// =============================================================================
int main()
{
    Vectmo vectmo;
    vectmo.setWorkingFile("vectmo_training_data");
    vectmo.createFile();

    std::string inputTrainText;

    std::cout << "\n==== Vectmo: Text Vectorization & Prediction ====\n" << std::endl;
    std::cout << "Enter text to train the model: ";
    std::getline(std::cin, inputTrainText);

    if (inputTrainText.empty())
    {
        std::cout << "[ERROR] No training text provided." << std::endl;
        return 1;
    }

    vectmo.pretrainModel(inputTrainText);

    // --- Prediction loop ---
    bool running = true;
    while (running)
    {
        std::string inputIncompleteText;

        std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
        std::cout << "║          Prediction Mode                ║" << std::endl;
        std::cout << "╚════════════════════════════════════════╝" << std::endl;
        std::cout << "Enter starter text (or 'quit' to exit): ";
        std::getline(std::cin, inputIncompleteText);

        if (inputIncompleteText == "quit" || inputIncompleteText == "exit")
        {
            std::cout << "\nExiting. Status Code: 0" << std::endl;
            break;
        }

        if (inputIncompleteText.empty())
        {
            std::cout << "[WARNING] Empty input. Try again." << std::endl;
            continue;
        }

        std::string predicted = vectmo.predictNextText(inputIncompleteText);

        std::cout << "\n┌─ PREDICTION ─────────────────────────┐" << std::endl;
        std::cout << "│ Input:  \"" << inputIncompleteText << "\"" << std::endl;
        std::cout << "│ Output: \"" << predicted << "\"" << std::endl;
        std::cout << "└───────────────────────────────────────┘" << std::endl;

        char choice;
        std::cout << "\nContinue? (Y/y/N/n): ";
        std::cin >> choice;
        std::cin.ignore();

        switch (choice)
        {
        case 'Y': case 'y':
            running = true;
            break;
        case 'N': case 'n':
            std::cout << "\nExiting. Status Code: 0" << std::endl;
            running = false;
            break;
        default:
            std::cout << "[INFO] Invalid choice. Continuing." << std::endl;
            running = true;
        }
    }

    return 0;
}
