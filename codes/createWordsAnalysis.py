import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import jieba
from sklearn.decomposition import PCA

# Chinese categories and English translations
category_translation = {
    "负面情绪": "Negative emotion",
    "生存痛苦": "Existential suffering",
    "自杀意图": "Suicide ideation",
    "自杀方法": "Suicide method",
    "自杀方式": "Suicide method",
    "执行过": "Execution state",
    "风险因素": "Risk factor",
    "过去时间": "The past",
    "现在": "The present",
    "未来": "The future",
    "确认": "Affirmation",
    "否认": "Negation",
}

# Cleaning text function
def clean_text(text):
    """
    Clean text: remove punctuation, standardize spaces
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Word segmentation function
def tokenize(text):
    """
    Use jieba to segment Chinese text
    """
    return list(jieba.cut(clean_text(text)))


data_path = "../data_final/finegrained_Suicide_dataall.tsv"
dictionary_path = "../data_final/dict_final/unique_words.csv"

data = pd.read_csv(data_path, sep="\t")
dictionary = pd.read_csv(dictionary_path)

# Clean the dictionary and convert to English categories
dictionary_dict = {clean_text(word): category_translation[cat] for word, cat in dictionary.values}

# Create a directory to save the charts
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Extract word frequency distribution
def compute_word_frequencies(texts, dictionary_dict):
    frequencies = {category: 0 for category in set(dictionary_dict.values())}
    for text in texts:
        words = tokenize(text)
        for word in words:
            if word in dictionary_dict:
                frequencies[dictionary_dict[word]] += 1
            else:
                print(f"Unmatched word: {word}")
    print("Frequencies:", frequencies)
    return frequencies
# Heatmap statistics function
def compute_heatmap_data(data, column, dictionary_dict):
    categories = set(dictionary_dict.values())
    labels = sorted(data["myLabel"].unique())
    matrix = {cat: [0] * len(labels) for cat in categories}
    for _, row in data.iterrows():
        words = tokenize(row[column])
        for word in words:
            if word in dictionary_dict:
                category = dictionary_dict[word]
                label = row["myLabel"]
                matrix[category][labels.index(label)] += 1
    return pd.DataFrame.from_dict(matrix, orient="index", columns=labels)



def plot_heatmap(data, title, filename):
    plt.figure(figsize=(16, 12))
    sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"fontsize": 18})


    plt.title(title, fontsize=28)
    plt.xlabel("Labels (0-10)", fontsize=22)
    plt.ylabel("Dictionary Categories", fontsize=22)

    # 让 x 轴标签居中
    plt.xticks(
        ticks=[i + 0.5 for i in range(len(data.columns))],
        labels=data.columns,
        rotation=0,
        fontsize=14
    )
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_bar_chart(frequencies, title, filename):
    plt.figure(figsize=(16, 12))
    plt.bar(frequencies.keys(), frequencies.values())
    plt.title(title, fontsize=16)
    plt.xlabel("Category", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_stacked_bar_chart(data, title, filename):
    data.plot(kind="bar", stacked=True, figsize=(16, 12))
    plt.title(title, fontsize=16)
    plt.xlabel("Dictionary Categories", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Labels", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def plot_pca(features, labels, title, filename):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(16, 12))
    for label in set(labels):
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Label {label}")
    plt.title(title, fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()


def analyze(data, column, suffix,suffix2):

    frequencies = compute_word_frequencies(data[column], dictionary_dict)
    plot_bar_chart(frequencies, f"Word Frequency Distribution ({suffix})", f"bar_chart_{suffix}.jpg")


    heatmap_data = compute_heatmap_data(data, column, dictionary_dict)
    plot_heatmap(heatmap_data, f"({suffix2}) Heatmap ({suffix})", f"heatmap_{suffix}.jpg")


analyze(data, "comment", "Oiginal User Posts", suffix2="a")


data["combined"] = data["comment"] + " " + data["explain"]
analyze(data, "combined", "User Posts with Inferred Explanation",suffix2="b")

print(f"Charts saved in {output_dir}.")
