import csv
import os
import re
import io
import gzip
import zipfile
from contextlib import contextmanager
import unicodedata
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

RAW_TEXT_COLUMNS = ('sentence', 'text', 'sents')
SEGMENTED_TEXT_COLUMNS = (
    'sentence_segmented',
    'text_segmented',
    'sentence_words',
    'text_words',
    'sentence_tokenized',
    'text_tokenized',
)

class VietnamesePreprocessor:
    def __init__(self, tokenizer_mode='auto'):
        if tokenizer_mode not in {'auto', 'whitespace', 'pretokenized'}:
            raise ValueError(
                "tokenizer_mode must be one of: auto, whitespace, pretokenized"
            )
        self.tokenizer_mode = tokenizer_mode
        self.acronyms = {
            ':)': 'colonsmile',
            ':(': 'colonsad',
            '@@': 'colonsurprise',
            '<3': 'colonlove',
            ':d': 'colonsmilesmile',
            ':3': 'coloncontemn',
            ':v': 'colonbigsmile',
            ':_': 'coloncc',
            ':p': 'colonsmallsmile',
            '>>': 'coloncolon',
            ':">': 'colonlovelove',
            '^^': 'colonhihi',
            ':': 'doubledot',
            ":'(": 'colonsadcolon',
            ':@': 'colondoublesurprise',
            'v.v': 'vdotv',
            '...': 'dotdotdot',
            '/': 'fraction',
            'c#': 'cshrap'
        }
        
        self.stopwords = set([
            'và', 'của', 'có', 'là', 'được', 'cho', 'với', 'từ', 'trong',
            'này', 'đó', 'để', 'một', 'những', 'các', 'không', 'thì', 'như',
            'đã', 'sẽ', 'khi', 'nếu', 'vì', 'mà', 'về', 'hoặc', 'hay',
            'đến', 'bởi', 'tại', 'theo', 'nên', 'nhưng', 'chỉ', 'nào', 'đây',
            'đấy', 'ở', 'ra', 'vào', 'lại', 'còn', 'cũng', 'rất', 'đều'
        ])
        self.vowel_table = [
            ['a', 'à', 'á', 'ả', 'ã', 'ạ'],
            ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
            ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
            ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
            ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
            ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
            ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
            ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
            ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
            ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
            ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
            ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'],
        ]
        self.vowel_to_ids = {}
        for vowel_index, forms in enumerate(self.vowel_table):
            for tone_index, char in enumerate(forms):
                self.vowel_to_ids[char] = (vowel_index, tone_index)

    def normalize_unicode(self, text):
        return unicodedata.normalize('NFC', text)

    def is_valid_vietnamese_word(self, word):
        vowel_positions = []
        for idx, char in enumerate(word):
            vowel_info = self.vowel_to_ids.get(char)
            if vowel_info is None:
                continue
            if vowel_positions and idx - vowel_positions[-1] != 1:
                return False
            vowel_positions.append(idx)
        return True

    def normalize_tone_word(self, word):
        if not self.is_valid_vietnamese_word(word):
            return word

        chars = list(word)
        tone_mark = 0
        vowel_positions = []
        qu_or_gi = False

        for idx, char in enumerate(chars):
            vowel_info = self.vowel_to_ids.get(char)
            if vowel_info is None:
                continue

            vowel_idx, tone_idx = vowel_info
            if vowel_idx == 9 and idx > 0 and chars[idx - 1] == 'q':
                chars[idx] = 'u'
                qu_or_gi = True
            elif vowel_idx == 5 and idx > 0 and chars[idx - 1] == 'g':
                chars[idx] = 'i'
                qu_or_gi = True

            if tone_idx != 0:
                tone_mark = tone_idx
                chars[idx] = self.vowel_table[vowel_idx][0]

            if not qu_or_gi or idx != 1:
                vowel_positions.append(idx)

        if not vowel_positions or tone_mark == 0:
            return ''.join(chars)

        target_idx = vowel_positions[0]
        if len(vowel_positions) > 1:
            for idx in vowel_positions:
                vowel_idx, _ = self.vowel_to_ids.get(chars[idx], (-1, -1))
                if vowel_idx in (4, 8):
                    target_idx = idx
                    break
            else:
                if len(vowel_positions) == 2:
                    target_idx = (
                        vowel_positions[0]
                        if vowel_positions[-1] == len(chars) - 1
                        else vowel_positions[1]
                    )
                else:
                    target_idx = vowel_positions[1]

        vowel_idx, _ = self.vowel_to_ids.get(chars[target_idx], (-1, -1))
        if vowel_idx != -1:
            chars[target_idx] = self.vowel_table[vowel_idx][tone_mark]

        return ''.join(chars)

    def replace_acronyms(self, text):
        sorted_acronyms = sorted(self.acronyms.items(), 
                                key=lambda x: len(x[0]), 
                                reverse=True)
        
        for acronym, replacement in sorted_acronyms:
            text = text.replace(acronym, f' {replacement} ')
        
        return text
    
    def preprocess(self, text, remove_stopwords=False, normalize_tone=True):
        text = self.normalize_unicode(text)
        text = self.replace_acronyms(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
    
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]
        if normalize_tone:
            tokens = [self.normalize_tone_word(t) for t in tokens]
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def preprocess_corpus(self, texts, remove_stopwords=False, normalize_tone=True):
        return [
            self.preprocess(text, remove_stopwords, normalize_tone)
            for text in texts
        ]

class CustomDatasetLoader:
    def __init__(self, data_dir='./data', tokenizer_mode='auto'):
        self.data_dir = data_dir
        self.preprocessor = VietnamesePreprocessor(tokenizer_mode=tokenizer_mode)
        self._warned_segmented_fallback = False

    def _extract_text(self, row, column_names):
        for column in column_names:
            value = row.get(column)
            if value is not None and value.strip():
                return value.strip()
        return None

    def _select_row_text(self, row):
        segmented_text = self._extract_text(row, SEGMENTED_TEXT_COLUMNS)
        raw_text = self._extract_text(row, RAW_TEXT_COLUMNS)
        tokenizer_mode = self.preprocessor.tokenizer_mode

        if tokenizer_mode == 'pretokenized':
            if segmented_text is not None:
                return segmented_text
            if raw_text is not None:
                if not self._warned_segmented_fallback:
                    print(
                        "Pretokenized mode requested but no segmented text column was found; "
                        "falling back to raw text columns."
                    )
                    self._warned_segmented_fallback = True
                return raw_text
            return None

        if tokenizer_mode == 'auto':
            return segmented_text if segmented_text is not None else raw_text

        return raw_text if raw_text is not None else segmented_text

    def _find_first_existing_file(self, filenames):
        for filename in filenames:
            candidate = os.path.join(self.data_dir, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _map_label(self, raw_value, mapping, label_name):
        key = str(raw_value).strip().lower()
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unsupported {label_name} label: {raw_value!r}")

    def _read_csv_split(self, csv_path):
        sentiment_map = {
            'negative': 0,
            'neg': 0,
            '0': 0,
            'neutral': 1,
            'neu': 1,
            '1': 1,
            'positive': 2,
            'pos': 2,
            '2': 2,
        }
        topic_map = {
            'lecturer': 0,
            '0': 0,
            'curriculum': 1,
            'training_program': 1,
            'training program': 1,
            'program': 1,
            '1': 1,
            'facility': 2,
            'facilities': 2,
            '2': 2,
            'others': 3,
            'other': 3,
            '3': 3,
        }
        texts = []
        sentiments = []
        topics = []

        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV file has no header: {csv_path}")

            for row in reader:
                text = self._select_row_text(row)

                if text is None:
                    raise ValueError(
                        f"Missing text column in {csv_path}. Expected one of: "
                        f"{RAW_TEXT_COLUMNS + SEGMENTED_TEXT_COLUMNS}"
                    )

                texts.append(text)
                sentiments.append(self._map_label(row.get('sentiment', ''), sentiment_map, 'sentiment'))
                topics.append(self._map_label(row.get('topic', ''), topic_map, 'topic'))

        return texts, sentiments, topics

    def _load_csv_splits(self):
        train_path = self._find_first_existing_file(('synthetic_train.csv', 'train.csv'))
        dev_path = self._find_first_existing_file(('synthetic_val.csv', 'val.csv', 'dev.csv'))
        test_path = self._find_first_existing_file(('synthetic_test.csv', 'test.csv'))

        if train_path is None:
            raise FileNotFoundError(
                "Could not find a training CSV in "
                f"{self.data_dir}. Checked: synthetic_train.csv, train.csv"
            )

        if dev_path is None and test_path is None:
            raise FileNotFoundError(
                "Could not find a validation/test CSV in "
                f"{self.data_dir}. Checked: synthetic_val.csv, val.csv, dev.csv, synthetic_test.csv, test.csv"
            )

        if dev_path is None:
            dev_path = test_path
            print(f"Validation CSV not found in {self.data_dir}; reusing {os.path.basename(test_path)} as dev split.")

        if test_path is None:
            test_path = dev_path
            print(f"Test CSV not found in {self.data_dir}; reusing {os.path.basename(dev_path)} as test split.")

        return (
            self._read_csv_split(train_path),
            self._read_csv_split(dev_path),
            self._read_csv_split(test_path),
        )
    
    def load_split(self, split_name):
        split_dir = os.path.join(self.data_dir, split_name)
        
        with open(os.path.join(split_dir, 'sents.txt'), 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f]

        with open(os.path.join(split_dir, 'sentiments.txt'), 'r', encoding='utf-8') as f:
            sentiments = [int(line.strip()) for line in f]
    
        with open(os.path.join(split_dir, 'topics.txt'), 'r', encoding='utf-8') as f:
            topics = [int(line.strip()) for line in f]
        
        assert len(texts) == len(sentiments) == len(topics)
        
        return texts, sentiments, topics
    
    def load_all_splits(self, task='sentiment', remove_stopwords=False, normalize_tone=True):
        has_legacy_layout = all(
            os.path.isdir(os.path.join(self.data_dir, split_name))
            for split_name in ('train', 'dev', 'test')
        )

        if has_legacy_layout:
            train_texts, train_sents, train_topics = self.load_split('train')
            dev_texts, dev_sents, dev_topics = self.load_split('dev')
            test_texts, test_sents, test_topics = self.load_split('test')
        else:
            (
                (train_texts, train_sents, train_topics),
                (dev_texts, dev_sents, dev_topics),
                (test_texts, test_sents, test_topics),
            ) = self._load_csv_splits()
        
        train_docs = self.preprocessor.preprocess_corpus(
            train_texts, remove_stopwords, normalize_tone
        )
        dev_docs = self.preprocessor.preprocess_corpus(
            dev_texts, remove_stopwords, normalize_tone
        )
        test_docs = self.preprocessor.preprocess_corpus(
            test_texts, remove_stopwords, normalize_tone
        )
        
        if task == 'sentiment':
            train_labels, dev_labels, test_labels = train_sents, dev_sents, test_sents
            num_classes = 3
        elif task == 'topic':
            train_labels, dev_labels, test_labels = train_topics, dev_topics, test_topics
            num_classes = 4
        else:
            raise ValueError(f"Not found: {task}")
        
        def avg_len(docs):
            return sum(len(d) for d in docs) / len(docs) if docs else 0

        print(f"\nLabel Distribution:")
        for split_name, labels in [('Train', train_labels), ('Dev', dev_labels), ('Test', test_labels)]:
            counts = defaultdict(int)
            for label in labels:
                counts[label] += 1
        
        return (train_docs, train_labels, dev_docs, dev_labels, 
                test_docs, test_labels, num_classes)


@contextmanager
def open_embedding_text(embedding_path):
    lower_path = embedding_path.lower()

    if lower_path.endswith('.gz'):
        with gzip.open(embedding_path, 'rt', encoding='utf-8', errors='ignore') as f:
            yield f
        return

    if lower_path.endswith('.zip'):
        with zipfile.ZipFile(embedding_path) as archive:
            members = [
                info for info in archive.infolist()
                if not info.is_dir()
            ]
            preferred_members = [
                info for info in members
                if info.filename.lower().endswith(('.txt', '.vec', '.emb'))
            ]
            candidates = preferred_members or members
            if not candidates:
                raise FileNotFoundError(f"No embedding file found inside archive: {embedding_path}")

            target = max(candidates, key=lambda info: info.file_size)
            print(f"Loading embeddings from archive member: {target.filename}")
            with archive.open(target, 'r') as raw_stream:
                with io.TextIOWrapper(raw_stream, encoding='utf-8', errors='ignore') as text_stream:
                    yield text_stream
        return

    with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
        yield f


def load_pretrained_embeddings(vocab, embedding_dim=300, embedding_path=None):
    vocab_size = len(vocab)
    
    embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    embeddings = embeddings * 0.01
    
    if embedding_path and os.path.exists(embedding_path):
        print(f"Loading embeddings from {embedding_path}...")
        found = set()
        
        with open_embedding_text(embedding_path) as f:
            for line in f:
                values = line.strip().split()
                if len(values) < embedding_dim + 1:
                    continue

                word = " ".join(values[:-embedding_dim])
                vector_values = values[-embedding_dim:]
                if not word:
                    continue

                if word in vocab:
                    vector = np.array(vector_values, dtype=np.float32)
                    if len(vector) == embedding_dim:
                        vocab_idx = vocab[word]
                        embeddings[vocab_idx] = vector
                        found.add(vocab_idx)
        
        print(f" Found {len(found)}/{vocab_size} vocabulary items in the pretrained embeddings.")
    else:
        print(f"Embedding path not provided or does not exist. Using random embeddings.")
    
    return embeddings


def load_glove_embeddings(vocab, embedding_dim=300, glove_path=None):
    return load_pretrained_embeddings(
        vocab,
        embedding_dim=embedding_dim,
        embedding_path=glove_path,
    )


class GFNDataset(Dataset):    
    def __init__(self, documents, labels, graph_constructor, corpus_graphs, 
                 p_neighbors=None):
        self.documents = documents
        self.labels = labels
        self.graph_constructor = graph_constructor
        self.corpus_graphs = corpus_graphs
        self.p_neighbors = p_neighbors
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        label = self.labels[idx]
        
        subgraphs = []
        node_indices_list = []
        edge_weights_list = []
        
        for corpus_adj in self.corpus_graphs:
            subgraph_adj, word_indices = self.graph_constructor.build_document_subgraph( doc, corpus_adj, p_neighbors=self.p_neighbors)
            
            if subgraph_adj is None or len(word_indices) == 0:
                subgraph_adj = np.array([[0.0]], dtype=np.float32)
                word_indices = [0]  
 
            src, dst = np.nonzero(subgraph_adj)
            
            if len(src) == 0:
                src = np.array([0])
                dst = np.array([0])
                edge_weights = np.array([1.0], dtype=np.float32)
            else:
                edge_weights = subgraph_adj[src, dst].astype(np.float32)
            
            g = dgl.graph((src, dst), num_nodes=len(word_indices))
            
            subgraphs.append(g)
            node_indices_list.append(torch.LongTensor(word_indices))
            edge_weights_list.append(torch.FloatTensor(edge_weights))
        
        return subgraphs, node_indices_list, edge_weights_list, label


def collate_fn(batch):
    all_subgraphs = [item[0] for item in batch]
    all_node_indices = [item[1] for item in batch]
    all_edge_weights = [item[2] for item in batch]
    labels = torch.LongTensor([item[3] for item in batch])
    
    batched_graphs = []
    batched_node_indices = []
    batched_edge_weights = []
    
    num_graphs = len(all_subgraphs[0]) 
    
    for graph_idx in range(num_graphs):
        graphs = [sample[graph_idx] for sample in all_subgraphs]
        node_indices = [sample[graph_idx] for sample in all_node_indices]
        edge_weights = [sample[graph_idx] for sample in all_edge_weights]
        
        batched_graph = dgl.batch(graphs)
        batched_node_idx = torch.cat(node_indices, dim=0)
        batched_edge_weight = torch.cat(edge_weights, dim=0)
        batched_graphs.append(batched_graph)
        batched_node_indices.append(batched_node_idx)
        batched_edge_weights.append(batched_edge_weight)
    
    return batched_graphs, batched_node_indices, batched_edge_weights, labels


def create_data_loaders(
    train_docs,
    train_labels,
    dev_docs,
    dev_labels,
    test_docs,
    test_labels,
    graph_constructor,
    corpus_graphs,
    batch_size=32,
    p_neighbors=None,
    num_workers=0,
    pin_memory=False,
    persistent_workers=None,
):

    train_dataset = GFNDataset(train_docs, train_labels, graph_constructor, corpus_graphs, p_neighbors)
    dev_dataset = GFNDataset(dev_docs, dev_labels, graph_constructor, corpus_graphs, p_neighbors)
    test_dataset = GFNDataset(test_docs, test_labels, graph_constructor, corpus_graphs, p_neighbors)

    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, dev_loader, test_loader
