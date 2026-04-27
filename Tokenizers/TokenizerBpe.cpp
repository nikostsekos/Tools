


#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>


#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>


using namespace std;


bool file_exists(const string& path)
{
    ifstream f(path.c_str());
    return f.good();
}



int train_bpe_model(const string& input_file,
                    const string& model_prefix,
                    int vocab_size)
{
    cout << "Training BPE model..." << endl;
    cout << "  input file:   " << input_file << endl;
    cout << "  model prefix: " << model_prefix << endl;
    cout << "  vocab size:   " << vocab_size << endl;


    stringstream options;

    options << "--input=" << input_file << " ";
    options << "--model_prefix=" << model_prefix << " ";
    options << "--vocab_size=" << vocab_size << " ";
    options << "--model_type=bpe" << " ";


    options << "--normalization_rule_name=nmt_nfkc" << " ";
    options << "--byte_fallback=true" << " ";
    options << "--split_by_unicode_script=true" << " ";
    options << "--split_by_whitespace=true" << " ";
    options << "--split_by_number=true" << " ";

    options << "--input_sentence_size=10000000" << " ";
    options << "--shuffle_input_sentence=true" << " ";
    options << "--max_sentence_length=16384" << " ";


    options << "--unk_id=0" << " ";
    options << "--bos_id=1" << " ";
    options << "--eos_id=2" << " ";
    options << "--pad_id=3";


    auto status = sentencepiece::SentencePieceTrainer::Train(options.str());

    if (!status.ok())
    {
        cout << "Training failed: " << status.ToString() << endl;
        return 1;
    }

    cout << "Training done!" << endl;
    cout << "  model file:  " << model_prefix << ".model" << endl;
    cout << "  vocab file:  " << model_prefix << ".vocab" << endl;

    return 0;
}

int tokenize_corpus(const string& corpus_file,
                    const string& model_path)
{
    sentencepiece::SentencePieceProcessor processor;

    auto load_status = processor.Load(model_path);

    if (!load_status.ok())
    {
        cout << "Failed to load model: "
             << load_status.ToString() << endl;
        return 1;
    }

    cout << "Model loaded from " << model_path << endl;
    cout << "Vocabulary size:   " << processor.GetPieceSize() << endl;
    cout << "\n--- Tokenizing corpus: " << corpus_file << " ---" << endl;


    ifstream in_file(corpus_file.c_str());

    if (!in_file.is_open())
    {
        cout << "Error: could not open corpus file: "
             << corpus_file << endl;
        return 1;
    }


    string line;
    int line_number  = 0;
    int total_tokens = 0;
    int total_lines  = 0;


    while (getline(in_file, line))
    {
        line_number = line_number + 1;

        // skip empty lines
        if (line.size() == 0)
        {
            continue;
        }


        vector<string> pieces;
        processor.Encode(line, &pieces);

        vector<int> ids;
        processor.Encode(line, &ids);


        cout << "\n[line " << line_number << "] " << line << endl;

        cout << "  pieces: ";
        for (size_t i = 0; i < pieces.size(); i++)
        {
            cout << "[" << pieces[i] << "] ";
        }
        cout << endl;

        cout << "  ids:    ";
        for (size_t i = 0; i < ids.size(); i++)
        {
            cout << ids[i] << " ";
        }
        cout << endl;

        cout << "  count:  " << ids.size() << " tokens" << endl;


        total_tokens = total_tokens + (int) ids.size();
        total_lines  = total_lines  + 1;
    }

    in_file.close();

   
    cout << "Lines tokenized: " << total_lines  << endl;
   
    cout << "Total tokens:    " << total_tokens << endl;

    if (total_lines > 0)
    {
        double avg = (double) total_tokens / (double) total_lines;
        cout << "Average tokens per line: " << avg << endl;
    }

    return 0;
}




void print_usage(const char* program_name)
{
    cout << "Usage:" << endl;
   
    cout << "  " << program_name      << " train    <corpus_file> <model_prefix> [vocab_size]" << endl;
    
    cout << "  " << program_name << " tokenize <text_file>   <model_prefix>" << endl;
    
    cout << endl;
    
    cout << "Examples:" << endl;
    
    cout << "  " << program_name << " train    big_greek_corpus.txt greek_bpe 32000" << endl;
    
    cout << "  " << program_name << " tokenize some_article.txt     greek_bpe" << endl;
}




int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    string command = argv[1];


  
    if (command == "train")
    {
        if (argc < 4)
        {
            print_usage(argv[0]);
            return 1;
        }

        string corpus_file  = argv[2];
        string model_prefix = argv[3];

        int vocab_size = 32000;
        if (argc >= 5)
        {
            vocab_size = atoi(argv[4]);
        }

        if (!file_exists(corpus_file))
        {
            cout << "Error: corpus file not found: "
                 << corpus_file << endl;
            return 1;
        }

        return train_bpe_model(corpus_file, model_prefix, vocab_size);
    }



    if (command == "tokenize")
    {
        if (argc < 4)
        {
            print_usage(argv[0]);
            return 1;
        }

        string text_file    = argv[2];
       
        string model_prefix = argv[3];
       
        string model_path   = model_prefix + ".model";

        if (!file_exists(text_file))
        {
            cout << "Error: text file not found: "
                 << text_file << endl;
            return 1;
        }

        if (!file_exists(model_path))
        {
            cout << "Error: model file not found: "<< model_path << endl;
            cout << "Did you forget to run '"<< argv[0] << " train ...' first?" << endl;
            return 1;
        }

        return tokenize_corpus(text_file, model_path);
    }


    cout << "Unknown command: " << command << endl;
    print_usage(argv[0]);
    return 1;
}

