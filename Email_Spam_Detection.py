"""Email spam detection. Submission name: Srivalli. Run: python Email_Spam_Detection.py [dataset.csv]
Install: pip install pandas numpy scikit-learn matplotlib joblib
Without a CSV, the public Apache SpamAssassin email corpus is downloaded.
For an actual email project, supply a labelled email CSV.
"""
import argparse
import re
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, ConfusionMatrixDisplay)


def clean(text):
    text = re.sub(r'<[^>]+>', ' ', str(text).lower())
    text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', ' EMAILTOKEN ', text)
    return ' '.join(text.split())


def predict_email(model, text):
    return 'Spam' if model.predict([clean(text)])[0] == 1 else 'Ham'


def main():
    parser = argparse.ArgumentParser(description='Spam/Ham classification')
    parser.add_argument('csv', nargs='?', help='Labelled email CSV file')
    parser.add_argument('--text-column')
    parser.add_argument('--label-column')
    parser.add_argument('--message', help='New email to classify')
    parser.add_argument('--show', action='store_true', help='Display charts')
    parser.add_argument('--output', default='spam_results')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.csv:
        try:
            df = pd.read_csv(args.csv)
        except UnicodeDecodeError:
            df = pd.read_csv(args.csv, encoding='latin-1')
        source = str(Path(args.csv).resolve())
    else:
        # Public email corpus from Apache SpamAssassin, downloaded on demand.
        import tarfile
        base = 'https://spamassassin.apache.org/old/publiccorpus/'
        records = []
        for archive, label in [('20030228_easy_ham.tar.bz2', 'ham'), ('20030228_spam.tar.bz2', 'spam')]:
            print('Downloading', archive)
            try:
                with urlopen(base + archive, timeout=60) as response:
                    content = response.read()
            except Exception as exc:
                raise SystemExit(f'Download failed: {exc}. Supply a local labelled email CSV instead.')
            with tarfile.open(fileobj=BytesIO(content), mode='r:bz2') as tar:
                for member in tar.getmembers():
                    if not member.isfile() or Path(member.name).name == 'cmds':
                        continue
                    from email import policy
                    from email.parser import BytesParser
                    msg = BytesParser(policy=policy.default).parsebytes(tar.extractfile(member).read())
                    body = msg.get_body(preferencelist=('plain', 'html'))
                    try:
                        content = body.get_content() if body else ''
                    except (LookupError, UnicodeError, KeyError):
                        content = ''
                    records.append({'text': str(msg.get('Subject', '')) + ' ' + str(content), 'label': label})
        df = pd.DataFrame(records)
        source = base + ' (20030228_easy_ham and 20030228_spam)'
    print('\nDATASET SOURCE:', source)
    print('Shape:', df.shape)
    df.info()
    print('Missing values:\n', df.isna().sum())
    cols = {str(c).strip().lower(): c for c in df.columns}
    def choose(explicit, candidates):
        if explicit:
            return explicit
        return next((cols[c] for c in candidates if c in cols), None)
    text_col = choose(args.text_column, ['email_text', 'message', 'text', 'body', 'email', 'v2'])
    label_col = choose(args.label_column, ['label', 'category', 'spam', 'class', 'target', 'v1'])
    if text_col not in df.columns or label_col not in df.columns:
        raise SystemExit(f'Cannot identify columns: {list(df.columns)}. Use --text-column and --label-column.')
    data = df[[text_col, label_col]].copy()
    data.columns = ['text', 'label']
    data = data.dropna()
    data['label'] = data.label.astype(str).str.strip().str.lower().map({'ham':0,'spam':1,'0':0,'1':1,'0.0':0,'1.0':1,'legitimate':0,'non-spam':0,'not spam':0,'false':0,'true':1})
    if data.label.isna().any():
        raise SystemExit('Unknown labels. Use ham/spam or 0/1.')
    data['label'] = data.label.astype(int)
    data['text'] = data.text.map(clean)
    data = data[data.text.ne('')].drop_duplicates()
    conflicts = data.groupby('text').label.nunique()
    data = data[~data.text.isin(conflicts[conflicts > 1].index)]
    print('\nFirst five cleaned records (text omitted for privacy):\n', data.assign(text=data.text.str.len()).rename(columns={'text':'characters'}).head())
    print('Cleaned shape:', data.shape, '\nClass counts:\n', data.label.value_counts())
    if len(data) < 10 or data.label.nunique() != 2 or data.label.value_counts().min() < 5:
        raise SystemExit('Provide at least 5 distinct messages per class and 10 total.')
    def save(name):
        plt.tight_layout(); plt.savefig(out / name, dpi=150)
        if args.show: plt.show()
        plt.close()
    data.label.map({0:'Ham',1:'Spam'}).value_counts().plot.bar(title='Class Distribution', rot=0)
    plt.ylabel('Messages'); save('class_distribution.png')
    for label, name in [(0,'Ham'),(1,'Spam')]:
        plt.hist(data.loc[data.label.eq(label),'text'].str.len(), bins=30, alpha=.55, label=name)
    plt.xlabel('Characters'); plt.ylabel('Messages'); plt.legend(); save('message_lengths.png')
    X_train, X_test, y_train, y_test = train_test_split(data.text, data.label, test_size=.2, random_state=42, stratify=data.label)
    for label, name in [(0,'ham'),(1,'spam')]:
        cv = CountVectorizer(stop_words='english')
        try:
            counts = cv.fit_transform(X_train[y_train.eq(label)])
            terms = pd.Series(counts.sum(axis=0).A1, index=cv.get_feature_names_out()).nlargest(15)
            terms.sort_values().plot.barh(title=f'Frequent {name} terms (training data)'); save(f'frequent_{name}.png')
        except ValueError:
            print('No usable frequent terms for', name)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=20000)
    train = vectorizer.fit_transform(X_train)
    test = vectorizer.transform(X_test)
    print('TF-IDF training/test shapes:', train.shape, test.shape)
    print('Training sparsity:', 1 - train.nnz / (train.shape[0]*train.shape[1]))
    models = {'Naive Bayes': MultinomialNB(), 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42), 'Linear SVM': LinearSVC(random_state=42)}
    # Choose using training-only cross-validation; reserve held-out data for evaluation.
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    folds = StratifiedKFold(n_splits=min(5, int(y_train.value_counts().min())), shuffle=True, random_state=42)
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, max_features=20000)), ('classifier', clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=folds, scoring='f1').mean()
        clf.fit(train, y_train)
        pred = clf.predict(test)
        rows.append({'Model':name,'CV F1':cv_f1,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred,zero_division=0),'Recall':recall_score(y_test,pred,zero_division=0),'F1':f1_score(y_test,pred,zero_division=0)})
        print('\n', name, '\n', classification_report(y_test,pred,labels=[0,1],target_names=['Ham','Spam'],zero_division=0))
    results = pd.DataFrame(rows).sort_values('CV F1', ascending=False)
    print('\nMODEL COMPARISON\n', results.to_string(index=False))
    results.to_csv(out/'model_comparison.csv', index=False)
    results.set_index('Model')[['Accuracy','Precision','Recall','F1']].plot.bar(rot=10, ylim=(0,1.05), title='Held-out test performance'); save('model_comparison.png')
    best = results.iloc[0]['Model']
    model = Pipeline([('tfidf', vectorizer), ('classifier', models[best])])
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), labels=[0,1], display_labels=['Ham','Spam'])
    plt.title(best); save('confusion_matrix.png')
    joblib.dump(model, out/'spam_model.joblib')
    (out/'dataset_source.txt').write_text(source, encoding='utf-8')
    print('\nSelected model:', best)
    for message in ['Congratulations! Claim your free prize now by clicking this link.', 'Please attend the project meeting tomorrow at 10 AM.'] + ([args.message] if args.message else []):
        print(f'{predict_email(model,message)}: {message}')
    print('Results saved to:', out.resolve())

if __name__ == '__main__':
    main()
