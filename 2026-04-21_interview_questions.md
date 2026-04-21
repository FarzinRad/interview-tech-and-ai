# سوالات مصاحبه فنی - ۲۰۲۶-۰۴-۲۱

---

## بخش اول: Data Science

---

### سوال ۱: Bias-Variance Tradeoff

**دسته:** آمار و یادگیری ماشین پایه  
**سطح:** Mid  
**چرا مهمه:** این مفهوم پایه‌ای‌ترین چالش در طراحی مدل‌های ML است و هر مهندس باید بتواند آن را هم به صورت نظری و هم عملی توضیح دهد.  
**زمان پاسخ:** ۱۰ تا ۱۵ دقیقه

**سوال:** Bias-Variance Tradeoff را توضیح دهید. چطور این دو خطا با هم در تعارض هستند؟ چه زمانی مدل دچار high bias یا high variance می‌شود و چطور هر کدام را برطرف می‌کنید؟

**پاسخ کامل:**

**مفهوم اصلی:**

خطای کلی یک مدل یادگیری ماشین را می‌توان به سه بخش تجزیه کرد:

```
Expected Error = Bias² + Variance + Irreducible Noise
```

**Bias** (تعصب) نشان‌دهنده خطای ناشی از فرضیات ساده‌کننده‌ای است که مدل در مورد داده دارد. وقتی مدل خیلی ساده‌تر از واقعیت است، نمی‌تواند پیچیدگی داده را یاد بگیرد و در نتیجه هم روی داده آموزشی و هم داده تست خطای بالایی دارد (underfitting).

**Variance** (واریانس) نشان‌دهنده حساسیت مدل به نوسانات کوچک در داده آموزشی است. مدل خیلی پیچیده، noise داده آموزشی را هم یاد می‌گیرد و روی داده جدید عملکرد ضعیفی دارد (overfitting).

**عمق فنی:**

از دیدگاه ریاضی، برای یک target function حقیقی f(x) و مدل آموزش‌دیده‌شده fˆ(x)، داریم:

```
E[(y - fˆ(x))²] = [Bias(fˆ(x))]² + Var(fˆ(x)) + σ²

Bias(fˆ(x)) = E[fˆ(x)] - f(x)
Var(fˆ(x)) = E[(fˆ(x) - E[fˆ(x)])²]
```

این یعنی وقتی Bias را کاهش می‌دهیم (مثلاً با پیچیده‌تر کردن مدل)، Variance افزایش می‌یابد و بالعکس. این trade-off بنیادی است.

**علائم High Bias:**
- Training error و Validation error هر دو بالا هستند
- شکاف کمی بین train و validation error وجود دارد
- افزودن داده بیشتر کمک زیادی نمی‌کند

**علائم High Variance:**
- Training error پایین است اما Validation error بالاست
- شکاف زیادی بین train و validation error وجود دارد
- افزودن داده بیشتر معمولاً کمک می‌کند

**راه‌حل‌های High Bias:**
۱. استفاده از مدل پیچیده‌تر (افزودن لایه‌ها، درجه polynomial بیشتر)
۲. کاهش regularization (کاهش λ در Ridge/Lasso)
۳. Feature Engineering برای افزودن ویژگی‌های مرتبط‌تر
۴. کاهش assumptions مدل

**راه‌حل‌های High Variance:**
۱. جمع‌آوری داده بیشتر
۲. افزایش regularization (L1, L2, Dropout)
۳. Feature Selection یا Dimensionality Reduction
۴. استفاده از Ensemble Methods (Bagging مثل Random Forest)
۵. Early Stopping در neural networks

**مثال واقعی:**

در یک پروژه پیش‌بینی قیمت خانه، اگر Linear Regression ساده با دو ویژگی استفاده کنیم، احتمالاً high bias داریم چون روابط غیرخطی را نادیده می‌گیرد. اگر یک درخت تصمیم عمیق بدون pruning استفاده کنیم، high variance خواهیم داشت. Random Forest با محدود کردن عمق درخت‌ها و averaging، این دو را متوازن می‌کند.

**اشتباهات رایج کاندیداها:**
- فکر می‌کنند که با داده بیشتر همیشه مشکل حل می‌شود (داده بیشتر فقط variance را کاهش می‌دهد، نه bias را)
- Regularization را فقط با Lasso/Ridge مرتبط می‌دانند و Dropout یا batch normalization را فراموش می‌کنند
- نمی‌توانند از روی learning curves تشخیص دهند کدام مشکل وجود دارد

**سوالات Follow-up:**
- آیا همیشه باید بین bias و variance trade-off کنیم؟ (نه — با داده بیشتر می‌توان هر دو را کاهش داد)
- Double descent phenomenon را توضیح دهید
- چطور cross-validation به تشخیص این مشکلات کمک می‌کند؟

---

### سوال ۲: روش‌های Feature Selection

**دسته:** Feature Engineering و Preprocessing  
**سطح:** Mid/Senior  
**چرا مهمه:** در دنیای واقعی با ده‌ها تا هزاران ویژگی مواجه‌ایم. انتخاب ویژگی‌های درست مستقیماً روی عملکرد، سرعت و تفسیرپذیری مدل تأثیر می‌گذارد.  
**زمان پاسخ:** ۱۰ تا ۱۵ دقیقه

**سوال:** انواع روش‌های Feature Selection را توضیح دهید. چه زمانی هر رویکرد مناسب‌تر است؟ چطور با multicollinearity کنار می‌آیید؟

**پاسخ کامل:**

**مفهوم اصلی:**

Feature Selection فرایند انتخاب زیرمجموعه‌ای از ویژگی‌های مرتبط برای ساخت مدل است. هدف اصلی: کاهش overfitting، بهبود دقت، کاهش زمان آموزش و افزایش تفسیرپذیری.

**سه دسته اصلی روش‌ها:**

**۱. Filter Methods (روش‌های فیلتر):**

این روش‌ها مستقل از مدل، بر اساس آمار آماری ویژگی‌ها را ارزیابی می‌کنند.

- **Variance Threshold:** ویژگی‌هایی با واریانس پایین (نزدیک به صفر) حذف می‌شوند. این ویژگی‌ها اطلاعات کمی دارند.
- **Correlation Coefficient:** محاسبه همبستگی پیرسون بین هر ویژگی و target. ویژگی‌های با |r| > threshold نگه داشته می‌شوند.
- **Chi-Square Test:** برای ویژگی‌های categorical، بررسی استقلال آماری بین ویژگی و target.
- **Mutual Information:** اندازه‌گیری وابستگی غیرخطی بین ویژگی و target. قوی‌تر از correlation چون روابط غیرخطی را هم می‌گیرد.
- **ANOVA F-Test:** برای مقایسه ویژگی‌های عددی در کلاس‌های مختلف.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=10)
X_new = selector.fit_transform(X, y)
```

**۲. Wrapper Methods (روش‌های پوشش‌دهنده):**

این روش‌ها از مدل ML برای ارزیابی زیرمجموعه ویژگی‌ها استفاده می‌کنند. گران‌تر ولی دقیق‌تر:

- **RFE (Recursive Feature Elimination):** مدل روی همه ویژگی‌ها آموزش می‌بیند، ضعیف‌ترین ویژگی حذف می‌شود، تکرار تا رسیدن به تعداد مطلوب.
- **Forward Selection:** از هیچ ویژگی‌ای شروع می‌کند و یکی یکی بهترین ویژگی را اضافه می‌کند.
- **Backward Elimination:** از همه ویژگی‌ها شروع می‌کند و بدترین را حذف می‌کند.

**۳. Embedded Methods (روش‌های تعبیه‌شده):**

Feature Selection در حین آموزش مدل انجام می‌شود:

- **Lasso (L1 Regularization):** کوچک کردن ضرایب کم‌اهمیت به صفر. طبیعتاً sparse solution تولید می‌کند.

```
Loss = MSE + λ * Σ|βᵢ|
```

- **Tree-based Feature Importance:** Random Forest و Gradient Boosting، بر اساس کاهش impurity یا permutation importance، اهمیت هر ویژگی را محاسبه می‌کنند.
- **ElasticNet:** ترکیب L1 و L2 — انعطاف بیشتر برای گروه‌هایی از ویژگی‌های correlated.

**مقابله با Multicollinearity:**

Multicollinearity زمانی رخ می‌دهد که دو یا چند ویژگی با هم همبستگی بالایی دارند (مثلاً قد به سانتیمتر و قد به اینچ). این مشکل:
- تخمین ضرایب رگرسیون را ناپایدار می‌کند
- Variance ضرایب را افزایش می‌دهد

**راه‌حل‌ها:**
۱. محاسبه VIF (Variance Inflation Factor): VIF > 10 نشانه مشکل جدی است
۲. حذف یکی از ویژگی‌های correlated
۳. PCA برای ایجاد ویژگی‌های مستقل
۴. Ridge Regression که با multicollinearity بهتر کنار می‌آید نسبت به OLS

**چه زمانی کدام روش:**
- داده‌های بزرگ با ویژگی‌های زیاد + سرعت مهم → Filter Methods
- عملکرد بهینه مهم + محاسبات کافی داریم → Wrapper Methods
- مدل‌های tree-based → Embedded (feature importance)
- مدل‌های خطی + sparse solution مطلوب → Lasso

**اشتباهات رایج کاندیداها:**
- Feature selection را روی کل داده (نه فقط train) انجام می‌دهند و data leakage ایجاد می‌کنند
- فکر می‌کنند بیشترین correlation با target، بهترین ویژگی است (mutual information قوی‌تر است)
- Multicollinearity در tree-based models را جدی می‌گیرند (درخت‌ها نسبت به آن مقاوم‌ترند)

**سوالات Follow-up:**
- تفاوت feature selection با dimensionality reduction چیست؟
- چطور feature importance در Random Forest و XGBoost محاسبه می‌شود؟
- SHAP values را توضیح دهید

---

### سوال ۳: مقابله با Imbalanced Datasets

**دسته:** Data Preprocessing و مدل‌سازی  
**سطح:** Mid/Senior  
**چرا مهمه:** در مسائل واقعی مثل تشخیص تقلب، تشخیص بیماری یا spam detection، کلاس‌های اقلیت اهمیت بیشتری دارند اما داده‌های کمتری دارند. این یکی از شایع‌ترین چالش‌های دنیای واقعی است.  
**زمان پاسخ:** ۱۰ تا ۱۵ دقیقه

**سوال:** با یک dataset که نسبت ۹۸:۲ بین کلاس‌ها دارد (تشخیص تقلب مالی) چطور کار می‌کنید؟ چه رویکردهایی برای handling کلاس‌های imbalanced وجود دارد؟ چه متریک‌هایی برای ارزیابی مناسب‌ترند؟

**پاسخ کامل:**

**مفهوم اصلی:**

وقتی یک کلاس در داده‌ها بسیار کمتر از کلاس‌های دیگر نمایندگی شده، مدل تمایل دارد همیشه کلاس اکثریت را پیش‌بینی کند — این در ظاهر accuracy بالایی (مثلاً ۹۸٪) می‌دهد اما از نظر business کاملاً بی‌فایده است.

**دسته‌بندی راه‌حل‌ها:**

**۱. Resampling Techniques:**

**Oversampling:**
- **Random Oversampling:** تکرار تصادفی نمونه‌های اقلیت. ساده اما ریسک overfitting دارد.
- **SMOTE (Synthetic Minority Over-sampling Technique):** ایجاد نمونه‌های مصنوعی با درون‌یابی بین نمونه‌های واقعی اقلیت. برای هر نمونه اقلیت، k نزدیک‌ترین همسایه پیدا می‌شود و نمونه جدیدی روی خط بین آن‌ها ایجاد می‌شود.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

- **ADASYN:** مشابه SMOTE اما نمونه‌های بیشتری در مناطقی که یادگیری سخت‌تر است ایجاد می‌کند.

**Undersampling:**
- **Random Undersampling:** حذف تصادفی نمونه‌های اکثریت. سریع اما اطلاعات از دست می‌رود.
- **Tomek Links:** حذف نمونه‌هایی از اکثریت که خیلی به مرز تصمیم نزدیکند. مرز را تمیزتر می‌کند.
- **NearMiss:** حذف نمونه‌هایی از اکثریت که به نمونه‌های اقلیت نزدیک‌ترند.

**ترکیب:** SMOTE + Tomek Links یا SMOTE + ENN معمولاً بهترین نتیجه را می‌دهد.

**۲. Algorithm-level Approaches:**

- **Class Weights:** تنظیم وزن کلاس‌ها در تابع loss. مدل جریمه بیشتری برای خطا در کلاس اقلیت می‌پردازد.

```python
from sklearn.svm import SVC
model = SVC(class_weight={0: 1, 1: 50})
# یا
model = SVC(class_weight='balanced')
```

- **Cost-Sensitive Learning:** تعریف ماتریس هزینه متفاوت برای انواع خطا.
- **Threshold Adjustment:** تغییر آستانه تصمیم از 0.5 به مقدار بهینه‌تر.

**۳. Ensemble Methods:**

- **BalancedBaggingClassifier:** در هر iteration، یک subset balanced از داده استفاده می‌شود.
- **EasyEnsemble:** ترکیب چند Undersampled dataset برای آموزش.
- **BalancedRandomForest:** مشابه Random Forest با resampling در هر درخت.

**متریک‌های مناسب:**

Accuracy برای imbalanced datasets گمراه‌کننده است. به جای آن:

- **Precision, Recall, F1-Score:** بر اساس business problem، اهمیت نسبی recall (کاهش false negative) و precision (کاهش false positive) متفاوت است.
- **ROC-AUC:** مقاوم در برابر imbalance، توانایی تفکیک مدل را نشان می‌دهد. اما در imbalance شدید گمراه‌کننده است.
- **PR-AUC (Precision-Recall AUC):** برای imbalanced datasets بهتر از ROC-AUC است.
- **G-Mean:** میانگین هندسی sensitivity و specificity.
- **Matthews Correlation Coefficient (MCC):** یکی از قوی‌ترین متریک‌ها برای imbalanced data.

```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**مثال واقعی - تشخیص تقلب:**

در پروژه fraud detection با نسبت ۹۸:۲:
۱. ابتدا baseline با class_weight='balanced' بسازید
۲. SMOTE روی training data اعمال کنید (نه test!)
۳. با PR-AUC و F1 برای کلاس fraud ارزیابی کنید
۴. Threshold را بر اساس هزینه business تنظیم کنید (false positive یا false negative گران‌تر است؟)

**اشتباهات رایج کاندیداها:**
- Resampling را روی کل داده (نه فقط train split) اعمال می‌کنند ← data leakage
- فقط به accuracy نگاه می‌کنند
- یادشان می‌رود که SMOTE برای continuous features کار می‌کند، نه categorical

**سوالات Follow-up:**
- تفاوت SMOTE و ADASYN چیست؟
- چه زمانی oversampling بهتر از undersampling است؟
- چطور threshold optimal را انتخاب می‌کنید؟

---

### سوال ۴: استراتژی‌های Cross-Validation

**دسته:** ارزیابی مدل  
**سطح:** Mid/Senior  
**چرا مهمه:** انتخاب اشتباه استراتژی cross-validation می‌تواند به تخمین بیش از حد خوش‌بینانه یا بدبینانه از عملکرد مدل منجر شود. این مهارت در مصاحبه‌های FAANG حتماً سنجیده می‌شود.  
**زمان پاسخ:** ۱۰ تا ۱۵ دقیقه

**سوال:** انواع مختلف cross-validation را توضیح دهید. چه زمانی هر روش مناسب است؟ چطور در time-series data باید CV انجام دهید؟ منظور از nested cross-validation چیست؟

**پاسخ کامل:**

**مفهوم اصلی:**

Cross-validation یک تکنیک برای تخمین عملکرد واقعی مدل روی داده‌های ندیده است. هدف: ارزیابی قابل اعتماد بدون نیاز به داده‌های جداگانه زیاد.

**انواع روش‌های CV:**

**۱. K-Fold Cross-Validation:**

داده به k بخش مساوی تقسیم می‌شود. در هر iteration یک بخش validation و بقیه training هستند. k-1 iteration تکرار می‌شود.

```python
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
```

انتخاب k: معمولاً k=5 یا k=10. k بزرگتر: bias کمتر، variance بیشتر، محاسبات بیشتر.

**۲. Stratified K-Fold:**

مانند K-Fold اما توزیع کلاس‌ها در هر fold حفظ می‌شود. **برای classification و imbalanced data ضروری است.**

**۳. Leave-One-Out Cross-Validation (LOOCV):**

k = تعداد نمونه‌ها. هر بار یک نمونه validation است. Unbiased اما محاسبات بسیار گران و variance بالاست. فقط برای dataset‌های خیلی کوچک.

**۴. Group K-Fold:**

وقتی داده گروه‌بندی دارد (مثلاً چند measurement از یک بیمار)، باید مطمئن شویم که نمونه‌های یک گروه در train و validation با هم نباشند. این data leakage را جلوگیری می‌کند.

**۵. Time Series Cross-Validation:**

مشکل اصلی: داده‌های زمانی ترتیب دارند. نمی‌توان به صورت تصادفی split کرد.

**Walk-Forward (Expanding Window):**
```
Train: [1..t]        → Validate: [t+1..t+h]
Train: [1..t+h]      → Validate: [t+h+1..t+2h]
...
```

**Sliding Window:**
```
Train: [1..w]        → Validate: [w+1..w+h]
Train: [2..w+1]      → Validate: [w+2..w+h+1]
...
```

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

**۶. Nested Cross-Validation:**

برای جلوگیری از optimistic bias در hyperparameter tuning. دو حلقه CV داریم:

- **Outer CV:** تخمین unbiased از عملکرد مدل (گزارش نهایی)
- **Inner CV:** انتخاب hyperparameter بهینه

```
For each outer fold:
    Tune hyperparameters with inner CV on training data
    Evaluate best model on outer validation fold
Report average of outer fold scores
```

بدون nested CV: اگر از همان داده برای tuning و evaluation استفاده کنیم، overly optimistic estimate می‌گیریم.

**مثال واقعی:**

در یک مسئله پیش‌بینی demand با داده سه سال:
- از Walk-forward validation با window سه ماهه استفاده کنید
- داده اول ۲ سال را برای اولین training fold نگه دارید
- هر بار یک ماه به جلو بروید و یک ماه predict کنید

**اشتباهات رایج کاندیداها:**
- استفاده از K-Fold برای time series ← data leakage از آینده به گذشته
- Feature engineering (scaling, normalization) قبل از CV اعمال می‌کنند ← leakage
- برای مقایسه مدل‌ها از test set استفاده می‌کنند (باید از CV استفاده کنند، test فقط برای گزارش نهایی)

**سوالات Follow-up:**
- چرا Shuffle در K-Fold برای time series غلط است؟
- Monte Carlo cross-validation چیست؟
- چطور CV را با Pipeline در sklearn پیاده‌سازی می‌کنید؟

---

### سوال ۵: آزمون فرضیه آماری در Data Science

**دسته:** آمار استنباطی  
**سطح:** Mid/Senior  
**چرا مهمه:** تصمیم‌گیری داده‌محور در شرکت‌های FAANG بدون درک عمیق آمار استنباطی ممکن نیست. A/B testing، تحلیل نتایج آزمایش‌ها، و اعتبارسنجی مدل همه به این مفاهیم نیاز دارند.  
**زمان پاسخ:** ۱۲ تا ۱۸ دقیقه

**سوال:** یک تیم Product می‌خواهد بداند آیا تغییر رنگ دکمه checkout از آبی به سبز، نرخ تبدیل را بهبود داده است. شما مسئول تحلیل آماری این A/B test هستید. مراحل کامل را توضیح دهید. p-value چیست و چه مشکلاتی دارد؟

**پاسخ کامل:**

**مفهوم اصلی:**

آزمون فرضیه یک چارچوب آماری است برای تصمیم‌گیری: آیا نتیجه مشاهده‌شده می‌تواند به طور تصادفی اتفاق افتاده باشد، یا نشانه یک اثر واقعی است؟

**مراحل کامل طراحی A/B Test:**

**گام ۱: تعریف فرضیه‌ها**
- **H₀ (فرضیه صفر):** تغییر رنگ دکمه تأثیری بر نرخ تبدیل ندارد (p_control = p_treatment)
- **H₁ (فرضیه جایگزین):** رنگ سبز نرخ تبدیل بالاتری دارد (p_treatment > p_control) — one-sided test

**گام ۲: تعیین پارامترهای آزمون قبل از شروع**

- **سطح معنا‌داری (α):** معمولاً ۰.۰۵. احتمال رد اشتباه H₀ وقتی درست است (Type I Error = False Positive)
- **Power (1-β):** معمولاً ۰.۸۰. احتمال تشخیص اثر واقعی (1 - Type II Error rate)
- **Minimum Detectable Effect (MDE):** حداقل بهبودی که business-relevant است (مثلاً ۲٪ بهبود نرخ تبدیل)

**گام ۳: محاسبه حجم نمونه**

```python
from scipy.stats import norm

def sample_size_calc(p_base, mde, alpha=0.05, power=0.80):
    p1 = p_base
    p2 = p_base + mde
    
    z_alpha = norm.ppf(1 - alpha)      # one-sided
    z_beta = norm.ppf(power)
    
    p_bar = (p1 + p2) / 2
    
    n = ((z_alpha * (2 * p_bar * (1 - p_bar))**0.5 + 
          z_beta * (p1*(1-p1) + p2*(1-p2))**0.5) / (p2 - p1))**2
    return int(n)
```

**گام ۴: اجرای آزمایش**

- تخصیص تصادفی کاربران به دو گروه
- جلوگیری از contamination (یک کاربر نباید هر دو version را ببیند)
- اجرا تا رسیدن به حجم نمونه محاسبه‌شده (نه "peek" کردن زود هنگام!)

**گام ۵: تحلیل نتایج**

برای مقایسه دو نسبت (conversion rate) از Two-proportion z-test استفاده می‌کنیم:

```python
from scipy.stats import chi2_contingency, proportions_ztest

# روش ۱: z-test
count = np.array([n_converted_treatment, n_converted_control])
nobs = np.array([n_treatment, n_control])
stat, p_value = proportions_ztest(count, nobs, alternative='larger')

# روش ۲: chi-square test (two-sided)
contingency_table = [[n_converted_t, n_not_t], 
                     [n_converted_c, n_not_c]]
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

**p-value چیست و چه مشکلاتی دارد؟**

p-value احتمال مشاهده نتیجه‌ای حداقل به اندازه extreme نتیجه مشاهده‌شده است، به فرض درستی H₀.

**تفسیر اشتباه رایج:** "p = 0.03 یعنی ۳٪ احتمال دارد H₀ درست باشد." این غلط است! p-value احتمال داده‌ها به فرض درستی H₀ است، نه احتمال H₀.

**مشکلات جدی p-value:**
۱. **Multiple Testing Problem:** اگر ۲۰ تست انجام دهیم، به احتمال زیاد یکی به طور تصادفی p < 0.05 می‌شود. راه‌حل: Bonferroni correction (α/m) یا Benjamini-Hochberg FDR
۲. **Peeking Problem:** اگر در حین آزمایش چندین بار نتیجه چک کنیم، false positive rate افزایش می‌یابد. راه‌حل: Sequential testing یا Bayesian approaches
۳. **Statistical vs Practical Significance:** p < 0.05 نمی‌گوید اثر مهم است. با n = یک میلیون، یک بهبود ۰.۰۰۱٪ هم significant می‌شود. **همیشه Effect Size (Cohen's d یا Relative Lift) گزارش دهید.**

**Confidence Interval بهتر از p-value:**

به جای p-value، Confidence Interval را گزارش دهید:
```
CI₉₅ = (p̂_t - p̂_c) ± z₀.₉₇₅ × SE
```

اگر CI شامل صفر نشود، نتیجه significant است — و همچنین اندازه و جهت اثر را نشان می‌دهد.

**مثال واقعی:**

در Meta، هزاران A/B test به صورت همزمان در حال اجراست. این نیازمند:
- **Experiment Platforms** مثل PlanOut یا Statsig
- **Multiple Testing Correction** خودکار
- **Variance Reduction** تکنیک‌ها مثل CUPED

**اشتباهات رایج کاندیداها:**
- قبل از رسیدن به sample size هدف، آزمایش را متوقف می‌کنند
- p-value را به عنوان احتمال درستی فرضیه تفسیر می‌کنند
- اثر novelty را در نظر نمی‌گیرند (کاربران به خاطر تازگی رفتار متفاوتی دارند)

**سوالات Follow-up:**
- تفاوت Bayesian A/B testing با Frequentist چیست؟
- چطور با network effects در A/B testing کنار می‌آیید؟
- CUPED چیست و چطور variance را کاهش می‌دهد؟

---

## بخش دوم: AI Engineering

---

### سوال ۶: طراحی MLOps Pipeline برای Production

**دسته:** AI Engineering / MLOps  
**سطح:** Senior/Staff  
**چرا مهمه:** مدل‌سازی تنها ۲۰٪ کار است. ۸۰٪ چالش در production، monitoring و نگهداری است. این سوال دانش end-to-end ML engineering را می‌سنجد.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** یک pipeline کامل MLOps برای یک مدل پیش‌بینی قیمت طراحی کنید. این pipeline باید شامل آموزش، ارزیابی، deployment، و monitoring باشد. چطور با model drift مقابله می‌کنید؟

**پاسخ کامل:**

**معماری کلی MLOps Pipeline:**

یک MLOps pipeline بالغ شامل چهار مرحله اصلی است که به صورت چرخه‌ای به هم متصل‌اند: Data → Training → Deployment → Monitoring → (feedback به Data).

**۱. Data Pipeline:**

```
Raw Data → Validation → Feature Engineering → Feature Store → Training Data
```

- **Data Validation:** Great Expectations یا TFX Data Validation برای بررسی schema، توزیع آماری و missing values
- **Feature Store:** Feast یا Tecton برای serving features یکسان در training و inference (جلوگیری از training-serving skew)
- **Data Versioning:** DVC یا Delta Lake برای نگهداری نسخه‌های داده

**۲. Model Training Pipeline:**

```python
# مثال با MLflow
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({"learning_rate": 0.01, "max_depth": 5})
    
    # Train model
    model = train_model(X_train, y_train, params)
    
    # Evaluate
    metrics = evaluate(model, X_val, y_val)
    mlflow.log_metrics(metrics)
    
    # Log model artifact
    mlflow.sklearn.log_model(model, "model")
```

- **Experiment Tracking:** MLflow، Weights & Biases، یا Neptune
- **Model Registry:** ثبت مدل با metadata کامل، versioning و stage management (Staging → Production)
- **Reproducibility:** Docker images برای environment، seed fixing، data versioning

**۳. Continuous Training (CT):**

مدل‌ها باید به صورت منظم retrain شوند:
- **Schedule-based:** هر هفته/ماه بدون در نظر گرفتن drift
- **Trigger-based:** وقتی drift شناسایی شد یا performance افت کرد
- **Event-based:** وقتی داده‌های لیبل‌دار جدید اضافه می‌شوند

**۴. Model Deployment:**

**Blue-Green Deployment:** دو محیط identical. ترافیک به یکباره به مدل جدید منتقل می‌شود. Rollback فوری ممکن است.

**Canary Release:** ترافیک به تدریج منتقل می‌شود (مثلاً ۱٪، ۵٪، ۲۵٪، ۱۰۰٪). ریسک کمتر اما پیچیده‌تر.

**Shadow Mode:** مدل جدید روی ترافیک واقعی اجرا می‌شود اما نتایجش استفاده نمی‌شود. برای مقایسه عملکرد قبل از go-live.

```yaml
# Kubernetes deployment با istio برای traffic splitting
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: model-v1
      weight: 90
    - destination:
        host: model-v2
      weight: 10
```

**۵. Monitoring — مقابله با Model Drift:**

دو نوع اصلی drift وجود دارد:

**Data Drift (Covariate Shift):** توزیع input features تغییر می‌کند.
```
PSI (Population Stability Index) = Σ (Actual% - Expected%) × ln(Actual%/Expected%)
PSI < 0.1: بدون تغییر
PSI 0.1-0.25: تغییر متوسط، بررسی لازم
PSI > 0.25: تغییر زیاد، retrain ضروری
```

**Concept Drift:** رابطه بین input و output تغییر می‌کند (دنیای واقعی تغییر کرده).

```python
# مثال monitoring با Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=production_data)
report.save_html("drift_report.html")
```

**Monitoring Dashboard باید شامل:**
- Data quality metrics (missing values، outliers، schema violations)
- Model performance metrics (در صورت وجود labels)
- Business KPIs (revenue impact، user satisfaction)
- Infrastructure metrics (latency، throughput، error rates)

**مثال واقعی — Uber Michelangelo:**

Uber یکی از بلوغ‌یافته‌ترین MLOps platformها را دارد. آن‌ها:
- Feature Store مرکزی برای اشتراک‌گذاری features بین تیم‌ها
- Automatic retraining با trigger-based monitoring
- A/B testing پیوسته برای مدل‌های جدید
- SLA-based deployment با auto-rollback

**اشتباهات رایج کاندیداها:**
- فقط از model performance monitoring صحبت می‌کنند و data monitoring را فراموش می‌کنند
- training-serving skew را نادیده می‌گیرند
- rollback strategy ندارند

**سوالات Follow-up:**
- تفاوت Continuous Training و Continuous Deployment در ML چیست؟
- Feature Store چه مشکلاتی را حل می‌کند؟
- چطور latency SLA را برای inference رعایت می‌کنید؟

---

### سوال ۷: سرویس‌دهی مدل در مقیاس بالا (Model Serving at Scale)

**دسته:** AI Engineering / Distributed Systems  
**سطح:** Senior/Staff  
**چرا مهمه:** در شرکت‌های بزرگ، مدل‌های ML باید میلیون‌ها درخواست در ثانیه را پاسخ دهند با latency زیر ۱۰۰ میلی‌ثانیه. این سوال توانایی طراحی سیستم‌های scalable را می‌سنجد.  
**زمان پاسخ:** ۱۵ تا ۲۵ دقیقه

**سوال:** یک سیستم inference برای یک مدل recommendation طراحی کنید که باید ۱۰۰ هزار QPS را با latency p99 زیر ۵۰ میلی‌ثانیه پشتیبانی کند. چه بهینه‌سازی‌هایی برای inference انجام می‌دهید؟

**پاسخ کامل:**

**معماری کلی Serving System:**

برای ۱۰۰K QPS با SLA سخت، باید یک سیستم چندلایه طراحی کرد:

```
User Request → Load Balancer → API Gateway
                                    ↓
                            Feature Serving Layer
                            (Redis/Feature Store)
                                    ↓
                            Model Server (GPU/CPU)
                                    ↓
                            Result Cache (Redis)
                                    ↓
                              Response to User
```

**۱. بهینه‌سازی مدل (Model Optimization):**

**Quantization:** کاهش دقت وزن‌ها از FP32 به INT8 یا INT4.
- حجم مدل تا ۴x کاهش می‌یابد
- Inference سرعت تا ۲-۴x افزایش می‌یابد
- با دقت کمی کاهش (معمولاً < 1% drop)

```python
import torch
from torch.quantization import quantize_dynamic

model_quantized = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Pruning:** حذف وزن‌های نزدیک به صفر. Unstructured pruning ← sparse matrix. Structured pruning ← سریع‌تر روی GPU.

**Knowledge Distillation:** آموزش یک مدل کوچک (student) از یک مدل بزرگ (teacher). Student رفتار teacher را تقلید می‌کند، نه فقط labels.

**ONNX Export:** تبدیل مدل به فرمت ONNX برای inference بهینه‌تر با ONNX Runtime.

**TensorRT:** بهینه‌سازی NVIDIA برای inference روی GPU:
- Layer fusion (ادغام چند عملیات)
- Kernel auto-tuning
- تا ۵x بهبود latency

**۲. Batching Strategies:**

**Dynamic Batching:** جمع‌آوری چند request و inference batch به جای تک‌تک:
```
Request 1 ─┐
Request 2 ─┤ → Batch [1,2,3] → GPU → Response 1,2,3
Request 3 ─┘
```

Triton Inference Server این کار را به صورت خودکار انجام می‌دهد با تنظیم `max_queue_delay_microseconds`.

**۳. Caching Strategies:**

**Result Cache:** کش کردن نتایج برای کاربران با request مشابه.
- برای recommendation: اگر user در آخرین X دقیقه result دریافت کرده، از cache استفاده کن
- TTL بر اساس freshness requirements

**Feature Cache:** ویژگی‌های از پیش محاسبه‌شده (مثلاً user embeddings) در Redis.

**۴. Horizontal Scaling:**

```
100K QPS → چند instance نیاز داریم؟
- هر GPU server: 5K QPS با 20ms latency
- برای 100K QPS: 20 GPU server
- با 20% overhead برای hot spare: 25 server
```

**Auto-scaling** بر اساس:
- CPU/GPU utilization
- Queue depth
- Request latency percentiles

**۵. Async vs Sync Inference:**

**Synchronous:** مناسب برای latency-sensitive applications (real-time recommendation)

**Asynchronous (Offline):** برای batch scoring، پیش‌محاسبه‌ی recommendations:
```
Kafka → Spark/Flink → Model Inference → Results DB
```

پیش‌محاسبه top-N recommendations برای هر کاربر شبانه، ذخیره در Redis. در request time، فقط lookup انجام می‌شود (< 1ms).

**۶. Infrastructure Stack:**

- **Model Server:** Triton Inference Server، TorchServe، Ray Serve
- **Load Balancer:** NGINX، Envoy
- **Feature Store:** Feast، Redis
- **Orchestration:** Kubernetes با GPU node pools
- **Monitoring:** Prometheus + Grafana، OpenTelemetry

**مثال واقعی — Netflix:**

Netflix از یک معماری hybrid استفاده می‌کند:
- Offline: پیش‌محاسبه candidate sets برای هر کاربر
- Online: یک ranking model سبک که روی candidates اجرا می‌شود
- Real-time features (trending) در context درخواست ترکیب می‌شوند

**اشتباهات رایج کاندیداها:**
- فقط از GPU scaling صحبت می‌کنند اما feature serving bottleneck را نادیده می‌گیرند
- از cold start problem غافل می‌شوند (برای کاربران جدید cache وجود ندارد)
- SLA را فقط برای average latency تعریف می‌کنند، نه p99

**سوالات Follow-up:**
- چطور A/B testing در real-time serving پیاده‌سازی می‌کنید؟
- Two-tower architecture برای recommendation چیست؟
- چطور GPU memory را بین چند مدل share می‌کنید؟

---

### سوال ۸: طراحی Feature Store

**دسته:** AI Engineering / Data Platform  
**سطح:** Senior/Staff  
**چرا مهمه:** Feature Store یکی از مهم‌ترین infrastructure components در MLOps است. بدون آن، تیم‌های ML وقت زیادی صرف بازسازی features می‌کنند و training-serving skew یک مشکل جدی است.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** Feature Store چیست و چه مشکلاتی را حل می‌کند؟ معماری یک Feature Store enterprise-grade را طراحی کنید. تفاوت online و offline store چیست؟

**پاسخ کامل:**

**مفهوم اصلی:**

Feature Store یک لایه data management تخصصی است که features (ویژگی‌های پردازش‌شده) را برای استفاده در training و serving یکسان می‌کند.

**مشکلات بدون Feature Store:**

۱. **Training-Serving Skew:** تیم A همان feature را به روش متفاوتی محاسبه می‌کند نسبت به تیم B. مدل روی داده متفاوتی serve می‌شود.
۲. **Duplication:** چند تیم همان feature را مستقل implement می‌کنند (اتلاف وقت و خطا).
۳. **Inconsistency:** feature در training با batch داده محاسبه می‌شود ولی در serving با real-time داده — نتیجه متفاوت است.
۴. **Versioning:** تغییر logic محاسبه feature بدون versioning، مدل‌های قدیمی را می‌شکند.

**معماری Feature Store:**

```
                    ┌─────────────────────────────┐
                    │         Feature Store         │
                    │                               │
Data Sources        │  ┌──────────┐ ┌───────────┐  │
(Kafka, S3, DB) ───►│  │  Feature  │ │  Feature  │  │
                    │  │Transform │ │ Registry  │  │
                    │  └──────────┘ └───────────┘  │
                    │       ↓             ↓         │
                    │  ┌──────────┐ ┌───────────┐  │
                    │  │ Offline  │ │  Online   │  │
                    │  │  Store   │ │   Store   │  │
                    │  │(S3/Hive) │ │  (Redis)  │  │
                    │  └──────────┘ └───────────┘  │
                    └─────────────────────────────┘
                          ↓                ↓
                    Training Jobs      Serving API
```

**Online Store در مقابل Offline Store:**

**Offline Store (Historical):**
- ذخیره‌سازی: columnar format (Parquet, Delta Lake) روی S3/HDFS
- مقیاس: petabytes
- latency: ثانیه‌ها تا دقیقه‌ها (batch processing)
- کاربرد: آموزش مدل، backfilling، analysis

**Online Store (Real-time):**
- ذخیره‌سازی: key-value store (Redis, DynamoDB, Cassandra)
- مقیاس: gigabytes (فقط آخرین مقدار features)
- latency: زیر ۱۰ میلی‌ثانیه
- کاربرد: real-time inference

**Point-in-time Correctness:**

یکی از مهم‌ترین چالش‌های Feature Store، جلوگیری از data leakage زمانی است. وقتی برای event در زمان T داریم، باید از features قبل از زمان T استفاده کنیم.

```python
# Feast point-in-time correct retrieval
feature_store = FeatureStore(repo_path=".")
training_df = feature_store.get_historical_features(
    entity_df=entity_df,  # شامل entity_id و event_timestamp
    features=[
        "user_stats:purchase_count_7d",
        "user_stats:avg_order_value",
    ]
).to_df()
```

**Feature Computation:**

**Batch Features:** محاسبه با Spark/Flink روی تاریخچه داده (مثلاً: تعداد خرید در ۳۰ روز گذشته)

**Streaming Features:** محاسبه real-time با Kafka + Flink (مثلاً: تعداد کلیک در ۵ دقیقه اخیر)

**On-demand Features:** محاسبه لحظه‌ای در زمان درخواست (مثلاً: فاصله بین کاربر و رستوران)

**مثال واقعی — Airbnb Zipline:**

Airbnb یکی از اولین شرکت‌هایی بود که Feature Store enterprise-grade ساخت. آن‌ها:
- بیش از ۱۰۰۰۰ feature دارند که بین ۱۰۰+ تیم share می‌شود
- Automatic backfilling برای features تاریخی
- Feature lineage tracking کامل

**اشتباهات رایج کاندیداها:**
- فکر می‌کنند Feature Store فقط برای ذخیره داده است (نه، اصلاً computation pipeline هم هست)
- Point-in-time correctness را نادیده می‌گیرند
- Online و offline store را مجزا نگه می‌دارند بدون sync mechanism

**سوالات Follow-up:**
- چطور feature freshness را guarantee می‌کنید؟
- Feature versioning چطور کار می‌کند؟
- برای streaming features، exactly-once semantics چطور پیاده‌سازی می‌کنید؟

---

### سوال ۹: LLM Fine-tuning و Serving در Production

**دسته:** AI Engineering / LLM Systems  
**سطح:** Senior/Staff  
**چرا مهمه:** با رشد LLMها، توانایی fine-tune کردن و deploy کردن آن‌ها در مقیاس تبدیل به یک مهارت ضروری برای AI Engineers شده است.  
**زمان پاسخ:** ۱۵ تا ۲۵ دقیقه

**سوال:** می‌خواهید یک LLM (مثلاً Llama-3 8B) را برای یک task خاص (customer support) fine-tune کنید و در production deploy کنید. مراحل کامل را توضیح دهید. تفاوت Full Fine-tuning، LoRA، و QLoRA را بگویید. چطور inference را بهینه می‌کنید؟

**پاسخ کامل:**

**مفهوم اصلی:**

Fine-tuning یعنی ادامه آموزش یک pre-trained model روی داده‌های domain-specific برای بهبود عملکرد در task خاص. برای LLMها این فرایند نیازمند منابع محاسباتی زیاد است مگر از روش‌های Parameter-Efficient Fine-Tuning (PEFT) استفاده کنیم.

**مقایسه روش‌های Fine-tuning:**

**Full Fine-tuning:**
- همه پارامترها آموزش می‌بینند
- بهترین نتیجه
- نیاز به GPU Memory زیاد: Llama-3 8B = 16 GB برای مدل + optimizer states ≈ 80+ GB
- نیاز به چند A100 80GB GPU

**LoRA (Low-Rank Adaptation):**

ایده اصلی: وزن‌های pre-trained را freeze می‌کنیم و دو ماتریس کوچک rank-decomposition اضافه می‌کنیم.

```
W_new = W_original + ΔW
ΔW = B × A    (rank r ≪ d)

W_original: d×d (frozen)
A: d×r (trainable)
B: r×d (trainable)
r = 8 یا 16 معمولاً کافی است
```

- فقط ۱-۵٪ پارامترها آموزش می‌بینند
- GPU memory: ~20-30GB برای 8B model
- عملکرد نزدیک به full fine-tuning در اکثر tasks

**QLoRA (Quantized LoRA):**

LoRA + 4-bit quantization برای base model. مدل اصلی با NF4 quantization ذخیره می‌شود.
- GPU memory: ~6-10GB برای 8B model (روی یک RTX 4090 امکان‌پذیر!)
- کمی کندتر از LoRA
- برای resource-constrained scenarios ایده‌آل

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**Data Preparation برای Customer Support:**

۱. **Instruction-tuning format:**
```json
{
  "instruction": "یک customer support agent هستی...",
  "input": "محصول من خراب شده، چطور تعویض کنم؟",
  "output": "با عرض تأسف از این مشکل..."
}
```

۲. **DPO (Direct Preference Optimization):** برای تنظیم tone و helpfulness با preference data.

**Inference Optimization برای Production:**

**KV Cache:** در LLM inference، attention keys و values برای tokens قبلی cache می‌شوند.

**Continuous Batching (Iteration-level scheduling):** به جای batch کردن کامل requests، در هر iteration، requests جدید به batch اضافه می‌شوند (vLLM این را پیاده‌سازی می‌کند).

**PagedAttention (vLLM):** مدیریت KV cache مشابه virtual memory در OS. از fragmentation جلوگیری می‌کند و throughput را ۲-۴x افزایش می‌دهد.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./fine-tuned-llama", 
           tensor_parallel_size=4,  # 4 GPU
           gpu_memory_utilization=0.95)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

**Speculative Decoding:** یک مدل کوچک (draft model) چند token پیش‌بینی می‌کند. مدل اصلی همه را در یک pass verify می‌کند. تا ۲x speedup در scenarios که acceptance rate بالاست.

**Tensor Parallelism:** تقسیم مدل روی چند GPU (وزن ماتریس‌ها به صورت horizontal تقسیم می‌شوند).

**مثال واقعی:**

برای deploy کردن Llama-3 8B fine-tuned:
- ۴ A100 40GB با tensor parallelism
- vLLM برای serving با continuous batching
- Throughput: ~۵۰۰۰ tokens/second
- Latency: ~۵۰ms برای اولین token (TTFT)

**اشتباهات رایج کاندیداها:**
- فکر می‌کنند LoRA همیشه بهتر از full fine-tuning است (برای domain adaptation گسترده، full fine-tuning بهتر است)
- Catastrophic forgetting را نادیده می‌گیرند
- فقط از throughput صحبت می‌کنند، TTFT (Time to First Token) را فراموش می‌کنند

**سوالات Follow-up:**
- Catastrophic forgetting چیست و چطور جلوگیری می‌کنید؟
- RLHF در مقابل DPO — تفاوت و مزایا؟
- چطور hallucination را در LLM production کاهش می‌دهید؟

---

### سوال ۱۰: طراحی سیستم تشخیص Anomaly در Real-time

**دسته:** AI Engineering / Streaming ML  
**سطح:** Senior/Staff  
**چرا مهمه:** anomaly detection در fraud detection، تشخیص خرابی سیستم، monitoring و امنیت سایبری کاربرد فراوان دارد. این سوال ترکیب ML و distributed systems را می‌سنجد.  
**زمان پاسخ:** ۱۵ تا ۲۵ دقیقه

**سوال:** یک سیستم real-time anomaly detection برای transaction‌های بانکی طراحی کنید. این سیستم باید تراکنش‌های مشکوک را با latency زیر ۱۰۰ms شناسایی کند. چه الگوریتم‌هایی مناسب است؟ چطور با concept drift کنار می‌آیید؟

**پاسخ کامل:**

**مفهوم اصلی:**

Anomaly detection هدفش شناسایی نقاط داده‌ای است که به طور قابل توجهی از pattern معمول منحرف می‌شوند. در تراکنش‌های بانکی، این شامل fraud، خطاهای سیستمی و رفتارهای غیرعادی کاربر است.

**معماری سیستم:**

```
Transaction → Kafka → Feature Engineering → ML Model → Decision Engine
                       (Flink)               (< 5ms)       (Rule + ML)
                           ↓                                     ↓
                      Feature Store                       Alert/Block
                       (Redis)                            + Audit Log
```

**لایه‌های تشخیص:**

برای کاهش false positive و افزایش coverage، از یک معماری چندلایه استفاده می‌کنیم:

**لایه ۱ — Rule-based (< 1ms):**
- تراکنش از کشور blacklist
- مبلغ بالاتر از حد مجاز
- سرعت غیرعادی (۳ تراکنش در ۱ دقیقه)

این لایه سریع و interpretable است اما adversarial attacks را نمی‌گیرد.

**لایه ۲ — Statistical Models (< 5ms):**

**Isolation Forest:** الگوریتمی بر پایه Random Forest. نقاط anomaly نیاز به تعداد کمتری برش (split) دارند.
```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.01, n_estimators=100)
clf.fit(normal_transactions)
anomaly_score = clf.score_samples(new_transaction)
```

**Z-score / Modified Z-score:** برای features عددی:
```
z = (x - μ) / σ
اگر |z| > 3: anomaly
```

**لایه ۳ — ML Model (< 20ms):**

**Gradient Boosting (XGBoost/LightGBM):** با features غنی:
- user history (آخر ۳۰ روز)
- device fingerprint
- merchant category
- time-based patterns
- velocity features (تعداد تراکنش در X دقیقه اخیر)

**AutoEncoder برای Unsupervised:**

شبکه‌ای که سعی می‌کند input را فشرده و بازسازی کند. برای تراکنش‌های normal، reconstruction error کم است. برای anomaly، error بزرگ است.

```python
class TransactionAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def anomaly_score(self, x):
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed)**2, dim=1)
```

**مقابله با Concept Drift:**

رفتار fraudsters تغییر می‌کند. مدل باید adapt شود.

**Online Learning:**
- River library برای incremental learning
- مدل با هر تراکنش جدید آپدیت می‌شود

**Windowed Retraining:**
- هر هفته مدل روی ۶ ماه داده اخیر retrain می‌شود
- Shadow deployment برای validation قبل از go-live

**Drift Detection:**
- Page-Hinkley Test یا ADWIN برای تشخیص تغییر توزیع
- وقتی drift شناسایی شد، retraining trigger می‌شود

**Feature Engineering Real-time:**

```python
# Velocity features با Redis
import redis
r = redis.Redis()

def get_velocity_features(user_id, current_time):
    key = f"user:{user_id}:transactions"
    
    # آخر ۱ ساعت
    one_hour_ago = current_time - 3600
    count_1h = r.zcount(key, one_hour_ago, current_time)
    
    # آخر ۱ روز
    one_day_ago = current_time - 86400
    sum_1d = r.zrangebyscore(key, one_day_ago, current_time)
    
    return {
        "tx_count_1h": count_1h,
        "tx_sum_1d": sum(float(x) for x in sum_1d)
    }
```

**مثال واقعی — Stripe Radar:**

Stripe از یک multi-layer approach استفاده می‌کند:
- بیش از ۱۰۰۰ signal برای هر تراکنش
- Adaptive machine learning که از feedback پذیرندگان یاد می‌گیرد
- Explainable AI: علت block کردن هر تراکنش نمایش داده می‌شود

**اشتباهات رایج کاندیداها:**
- فقط supervised learning را پیشنهاد می‌دهند (labels همیشه موجود نیست)
- از latency budget برای هر component غافل می‌شوند
- False positive cost را در نظر نمی‌گیرند (blocking کاربر خوب هم هزینه دارد)

**سوالات Follow-up:**
- چطور false positive rate را بدون افزایش false negative کاهش می‌دهید؟
- Graph Neural Networks در fraud detection چه نقشی دارند؟
- چطور با adversarial attacks مقابله می‌کنید؟

---

## بخش سوم: DL/ML System Design

---

### سوال ۱۱: طراحی سیستم Recommendation Engine

**دسته:** ML System Design  
**سطح:** Senior/Staff  
**چرا مهمه:** Recommendation systems قلب تپنده Netflix، Spotify، Amazon و YouTube هستند. این سوال توانایی طراحی سیستم ML end-to-end را در مقیاس بزرگ می‌سنجد.  
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک Recommendation System برای یک پلتفرم محتوا (مشابه Netflix با ۲۰۰ میلیون کاربر و ۱۰ هزار عنوان) طراحی کنید. Requirements: real-time personalization، cold start handling، و قابلیت A/B testing.

**پاسخ کامل:**

**Clarification Questions (که در مصاحبه باید بپرسید):**
- آیا هدف maximize engagement است یا satisfaction (اینها می‌توانند متفاوت باشند)؟
- چه نوع محتوایی؟ (فیلم، سریال، انیمیشن)
- آیا explicit ratings داریم یا implicit (watch history)?
- Latency requirement چیست؟

**معماری کلی — دو مرحله‌ای:**

سیستم‌های recommendation در مقیاس بزرگ از معماری دو مرحله‌ای (Two-stage) استفاده می‌کنند:

```
۲۰۰M کاربران × ۱۰K محتوا = ۲ تریلیون ترکیب ممکن

مرحله ۱: Retrieval/Candidate Generation
    ۲ تریلیون → ۱۰۰-۱۰۰۰ candidate
    
مرحله ۲: Ranking
    ۱۰۰۰ candidate → Top-10 recommendation
```

**مرحله ۱: Candidate Generation**

**Collaborative Filtering — Matrix Factorization:**

هر کاربر و محتوا به یک embedding vector در یک فضای latent mapping می‌شوند. پیش‌بینی: امتیاز = uᵢ · vⱼ (dot product).

```python
# Two-tower model
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim=64):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(user_dim, 128),
            nn.Linear(128, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(item_dim, 128),
            nn.Linear(128, embed_dim)
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_tower(user_ids)
        item_emb = self.item_tower(item_ids)
        return torch.sum(user_emb * item_emb, dim=1)
```

**ANN (Approximate Nearest Neighbor) Search:**

به جای dot product با همه ۱۰K محتوا، user embedding را در یک ANN index جستجو می‌کنیم.
- **FAISS** (Facebook AI Similarity Search) با HNSW algorithm
- IVF-PQ برای memory efficiency
- latency: < 5ms برای ۱۰ میلیون آیتم

**مرحله ۲: Ranking**

یک مدل پیچیده‌تر (معمولاً Deep Neural Network) که features غنی‌تری دارد:
- User context (زمان روز، device، جلسه اخیر)
- Item features (genre، language، طول، تازگی)
- Interaction features (آیا دوستان کاربر دیده‌اند؟)
- Historical engagement (click rate، completion rate)

**DNN Ranking:**
```python
class RankingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        return self.dense(features)
```

**Cold Start Problem:**

**برای کاربران جدید:**
- نمایش popular items که segment مشابه (age، location) دوست دارند
- Onboarding: پرسیدن چند ژانر مورد علاقه
- Contextual bandits برای exploration سریع

**برای محتوای جدید:**
- Content-based filtering با metadata (genre، cast، synopsis embedding)
- Cross-selling: اگر بازیگر مشهور دارد، به کاربرانی که آن بازیگر را دوست دارند نشان بده
- Exploration pool: ۵-۱۰٪ از recommendations برای discover محتوای جدید

**Diversity و Freshness:**

صرفاً ranking score کافی نیست. باید:
- **Diversity:** جلوگیری از نمایش ۱۰ فیلم مشابه (MMR: Maximal Marginal Relevance)
- **Freshness:** تازه‌ترین محتوا وزن بیشتری داشته باشد
- **Serendipity:** محتوایی که کاربر انتظار ندارد اما دوست خواهد داشت

**Metrics:**

- **Offline:** NDCG@K، Hit Rate@K، Coverage
- **Online:** Click-through Rate، Watch Time، Completion Rate، Return Rate

**اشتباهات رایج کاندیداها:**
- فقط از Collaborative Filtering صحبت می‌کنند بدون در نظر گرفتن scalability
- Cold start را فراموش می‌کنند
- فقط engagement را optimize می‌کنند نه user satisfaction (echo chamber problem)

**سوالات Follow-up:**
- چطور filter bubble را کاهش می‌دهید؟
- Session-based recommendation چیست؟
- چطور multi-armed bandit در recommendation استفاده می‌کنید؟

---

### سوال ۱۲: طراحی سیستم Search Ranking

**دسته:** ML System Design / Information Retrieval  
**سطح:** Senior/Staff  
**چرا مهمه:** جستجو قلب محصولات Google، Amazon، LinkedIn و Airbnb است. این سوال درک عمیق از Learning to Rank و NLP در مقیاس بزرگ را می‌سنجد.  
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک سیستم جستجوی محصول برای یک فروشگاه آنلاین (مانند Amazon با ۳۰۰ میلیون محصول) طراحی کنید. باید هم relevance و هم business metrics (فروش) را optimize کند.

**پاسخ کامل:**

**Clarification Questions:**
- آیا semantic search یا keyword search؟
- چه زبان‌هایی پشتیبانی شود؟
- Latency requirement (معمولاً < 200ms end-to-end)?
- آیا personalization نیاز است؟

**معماری چندلایه:**

```
Query → Query Understanding → Retrieval → Ranking → Business Logic → Results
```

**لایه ۱: Query Understanding**

قبل از هر چیز، باید query را درک کنیم:

- **Spell correction:** "lapptop" → "laptop"
- **Query expansion:** "laptop" → "laptop, notebook computer"
- **Intent classification:** navigational / informational / transactional
- **Entity recognition:** "iPhone 15 Pro 256GB" ← برند، مدل، storage
- **Query rewriting:** با مدل‌های NLP مثل BERT

**لایه ۲: Retrieval (Candidate Generation)**

**Inverted Index (BM25):**

```
BM25(q, d) = Σ IDF(tᵢ) × (tf(tᵢ,d) × (k₁+1)) / (tf(tᵢ,d) + k₁ × (1 - b + b×|d|/avgdl))
```

این سریع (milliseconds برای miilions of documents) و interpretable است.

**Dense Retrieval (Semantic Search):**

Bi-encoder: query و document هر کدام مستقل encode می‌شوند.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

query_embedding = model.encode(query)
# ANN search با FAISS
scores, indices = faiss_index.search(query_embedding, k=1000)
```

**Hybrid Retrieval:** ترکیب BM25 و Dense Retrieval با Reciprocal Rank Fusion:
```
RRF(d) = Σ 1/(k + rank_i(d))
```

**لایه ۳: Learning to Rank (LTR)**

**Pointwise:** هر document را مستقل score می‌دهیم (regression/classification).

**Pairwise (RankNet):** مدل یاد می‌گیرد کدام document بهتر از دیگری است.

**Listwise (LambdaMART):** کل ranked list را با optimize کردن NDCG مستقیماً آموزش می‌دهیم.

```python
import lightgbm as lgb

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "num_leaves": 63,
    "learning_rate": 0.05
}

model = lgb.train(params, train_data, num_boost_round=500)
```

**Features برای Ranking:**

- **Query-Document Match:** BM25 score، semantic similarity، title match، exact query match
- **Document Quality:** average rating، number of reviews، return rate
- **Popularity:** sales rank، click-through rate، conversion rate
- **Freshness:** days since last update
- **Personalization:** user history، similar purchases، category preference

**لایه ۴: Business Logic**

بعد از ranking خالص ML، business rules اعمال می‌شود:
- **Promoted listings:** sponsored products با label واضح
- **Inventory filter:** محصولات out of stock پایین‌تر می‌آیند
- **Policy filter:** محصولات prohibited حذف می‌شوند
- **Diversity:** از هر seller حداکثر N محصول

**Training Data — Click Logs:**

```
query: "wireless headphones"
position_1: Sony WH-1000XM5 → clicked ✓
position_2: Apple AirPods → not clicked ✗
position_3: Bose QuietComfort → clicked ✓
```

**Position Bias:** کاربران item های بالاتر را بیشتر click می‌کنند صرفاً به خاطر موقعیت. باید با **Inverse Propensity Scoring (IPS)** این bias را تصحیح کنیم.

**Evaluation:**

- **Offline:** NDCG@10، MRR، MAP
- **Online:** CTR، Conversion Rate، GMV، Search Abandonment Rate

**مثال واقعی:**

Airbnb از یک ranking model استفاده می‌کند که ۱۰۰+ feature دارد. آن‌ها کشف کردند که لیستینگ‌های با عکس‌های با کیفیت بالاتر، حتی با کنترل سایر عوامل، نرخ رزرو بالاتری دارند ← این به عنوان feature اضافه شد.

**اشتباهات رایج کاندیداها:**
- فقط از keyword matching صحبت می‌کنند و semantic search را نادیده می‌گیرند
- Position bias را در training data نادیده می‌گیرند
- Trade-off بین relevance و revenue را فراموش می‌کنند

**سوالات Follow-up:**
- BERT-based cross-encoder در مقابل bi-encoder: trade-offs؟
- چطور با tail queries (queries نادر) کار می‌کنید؟
- Counterfactual evaluation چیست؟

---

### سوال ۱۳: طراحی سیستم تشخیص محتوای مضر (Content Moderation)

**دسته:** DL System Design / Trust & Safety  
**سطح:** Senior/Staff  
**چرا مهمه:** هر پلتفرم اجتماعی بزرگ (Meta، YouTube، Twitter/X) به سیستم‌های moderation در مقیاس عظیم نیاز دارد. این سوال ترکیب Computer Vision، NLP، و Human-in-the-loop را می‌سنجد.  
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک سیستم content moderation برای یک پلتفرم اجتماعی با ۵۰۰ میلیون پست در روز (متن، تصویر، ویدیو) طراحی کنید. هدف: تشخیص hate speech، violence، spam، و محتوای جنسی. چطور با false positives و cultural context کنار می‌آیید؟

**پاسخ کامل:**

**Clarification Questions:**
- چه نوع محتوایی: text only، images، videos، یا ترکیبی؟
- Latency: real-time (blocking at upload) یا near-real-time?
- چند زبان پشتیبانی می‌شود؟
- آیا appeal mechanism وجود دارد؟

**چالش‌های اصلی:**

۱. **Scale:** ۵۰۰ میلیون پست در روز = ~۵۷۸۰ پست در ثانیه
۲. **Multilingual:** صدها زبان با nuances فرهنگی متفاوت
۳. **False Positive Cost:** حذف محتوای مشروع به کاربران آسیب می‌زند
۴. **Adversarial:** بد actors سیستم را دور می‌زنند
۵. **Context:** یک عکس ممکن است در یک context مجاز و در دیگری مضر باشد

**معماری چندلایه:**

```
Content Upload
      ↓
┌─────────────────────────────────────────┐
│          Tiered Decision Making          │
│                                         │
│  Tier 1: Hash Matching (< 1ms)          │
│  Tier 2: ML Classifier (< 50ms)         │
│  Tier 3: Human Review (hours)           │
└─────────────────────────────────────────┘
      ↓
Decision: Allow / Remove / Restrict / Escalate
```

**لایه ۱: Hash Matching (Exact & Near-duplicate)**

- **PhotoDNA:** Microsoft's perceptual hash برای CSAM detection. Hash محتوای شناخته‌شده مضر را در یک database نگه می‌داریم.
- **SimHash:** برای near-duplicate text
- این لایه: O(1) lookup، بسیار سریع، false positive تقریباً صفر

**لایه ۲: ML Classification**

**Text Moderation:**

```python
# Fine-tuned multilingual BERT
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=5  # clean, hate, violence, spam, adult
)

# Multi-label: یک پست می‌تواند چند نوع violation داشته باشد
outputs = model(input_ids, attention_mask)
probs = torch.sigmoid(outputs.logits)
```

**Image Moderation:**

- Fine-tuned ResNet یا ViT برای تصویر
- Object detection برای explicit content
- Context embedding: متن همراه تصویر را هم در نظر بگیریم

**Video Moderation:**

- Frame sampling (هر N frame یک بار بررسی شود)
- Audio transcription + text moderation
- Temporal context: یک frame ممکن است بی‌خطر باشد اما در context ویدیو مضر باشد

**Multi-modal Fusion:**

```python
class MultiModalModerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-multilingual")
        self.image_encoder = ViT.from_pretrained("google/vit-base-patch16-224")
        self.fusion = nn.Linear(768 + 768, 5)  # 5 violation categories
    
    def forward(self, text_input, image_input):
        text_features = self.text_encoder(**text_input).pooler_output
        image_features = self.image_encoder(image_input).pooler_output
        combined = torch.cat([text_features, image_features], dim=1)
        return self.fusion(combined)
```

**لایه ۳: Human Review**

نه همه موارد را باید انسان‌ها بررسی کنند — این scale‌پذیر نیست. فقط:
- موارد با confidence پایین (0.4 < score < 0.7)
- Appeals از کاربران
- Samples تصادفی برای quality assurance

**Prioritization Queue:**
- Viral content (engagement بالا) اولویت بالاتر دارد
- Reported content توسط کاربران
- Content از حساب‌های با history violation

**مقابله با Cultural Context:**

یک سیستم جهانی باید بداند که:
- یک عبارت در یک فرهنگ توهین است اما در دیگری خیر
- Context سیاسی متفاوت است (انتقاد دولت در بعضی کشورها illegal است)
- Satire و irony سخت است برای مدل تشخیص دهد

**رویکردها:**
- Regional models با داده محلی
- Cultural sensitivity layers
- Human reviewer‌های با تخصص regional
- Localized policy enforcement

**Adversarial Robustness:**

Bad actors از ترفندهایی مانند:
- تغییر حروف ("h@te sp33ch")
- استفاده از زبان coded
- Image manipulation (پیکسل‌بندی)
- تصاویر text-embedded

برای مقابله:
- Adversarial training
- OCR روی تصاویر برای استخراج متن
- Character-level models که به typos مقاوم‌اند

**Metrics:**

- **Precision:** از آنچه حذف کردیم، چقدر واقعاً مضر بود؟
- **Recall:** از تمام محتوای مضر، چقدر گرفتیم؟
- **Prevalence:** نسبت محتوای مضر قبل از action
- **Appeals Rate:** چند درصد از removals اشتباه بودند؟

**مثال واقعی:**

Meta Oversight Board یک لایه اضافه برای بررسی تصمیمات مهم است. آن‌ها از ترکیب AI + human reviewers استفاده می‌کنند و ~۵ میلیون پیس محتوا را در روز بررسی می‌کنند.

**اشتباهات رایج کاندیداها:**
- فکر می‌کنند یک مدل global می‌تواند همه فرهنگ‌ها را cover کند
- Human review را فراموش می‌کنند
- Adversarial robustness را نادیده می‌گیرند

**سوالات Follow-up:**
- چطور با zero-day violating content (نوع جدید که مدل ندیده) کنار می‌آیید؟
- Explainability در moderation decision چقدر مهم است؟
- چطور reviewer burnout را مدیریت می‌کنید؟

---

### سوال ۱۴: طراحی سیستم پیش‌بینی تقاضا (Demand Forecasting)

**دسته:** DL/ML System Design / Time Series  
**سطح:** Senior/Staff  
**چرا مهمه:** Demand forecasting در Uber، Lyft، Amazon، و DoorDash بیلیون‌ها دلار صرفه‌جویی ایجاد می‌کند. این سوال ترکیب time series، deep learning، و business impact را می‌سنجد.  
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک سیستم Demand Forecasting برای یک پلتفرم ride-hailing (مانند Uber) طراحی کنید. باید تقاضا را در ۳۰ دقیقه آینده برای هر منطقه جغرافیایی پیش‌بینی کند تا drivers را از قبل positioning کند.

**پاسخ کامل:**

**Clarification Questions:**
- گرانولاریتی چیست؟ (هر h3 cell در hexagonal grid مثل Uber H3)
- Horizon پیش‌بینی: ۳۰ دقیقه vs چند ساعت؟
- آیا weather, events, و holidays را داریم؟
- Update frequency: هر چند دقیقه مدل predict می‌کند؟

**Problem Definition:**

```
Input: تاریخچه ۶ ماه demand در هر cell، external features
Output: تعداد درخواست‌های پیش‌بینی‌شده برای هر cell در t+30min
Scale: ۱۰K+ cells در یک شهر بزرگ
```

**معماری داده:**

**زمانبندی پیش‌بینی:**
```
هر ۵ دقیقه:
1. Collect latest demand data
2. Update features
3. Run inference
4. Push predictions to driver-dispatch system
```

**Feature Engineering:**

**Time-based Features:**
- ساعت روز (sin/cos encoding برای cyclical patterns)
- روز هفته، ماه، فصل
- آیا روز تعطیل است؟
- زمان از آخرین رویداد بزرگ

**Spatial Features:**
- Cell ID embedding
- Spatial aggregation: demand در cells همسایه
- Distance to city center، airport، مراکز خرید

**Contextual Features:**
- دمای هوا، بارندگی
- رویدادهای بزرگ (کنسرت، ورزش)
- ترافیک real-time

**Demand History:**
- تقاضای ۱ ساعت اخیر (resolution دقیقه‌ای)
- تقاضای یک هفته پیش همین زمان
- تقاضای یک ماه پیش همین زمان

**مدل‌های مناسب:**

**۱. Classical — SARIMA/SARIMAX:**

مناسب برای single time series ساده با seasonality. اما برای ۱۰K+ series به صورت موازی گران است.

**۲. Tree-based — LightGBM/XGBoost:**

```python
import lightgbm as lgb

# هر cell به صورت مستقل یا با cell_id به عنوان feature
model = lgb.LGBMRegressor(
    objective='huber',  # مقاوم در برابر outlier
    num_leaves=63,
    n_estimators=500
)

# Features: lag features, time features, external features
model.fit(X_train, y_train)
```

این approach خوب است اما spatial correlation را نادیده می‌گیرد.

**۳. Deep Learning — Temporal Fusion Transformer (TFT):**

State-of-the-art برای time series forecasting. ترکیب:
- LSTM برای sequential patterns
- Multi-head attention برای long-range dependencies
- Variable selection network برای اهمیت features

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Configuration
dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="demand",
    group_ids=["cell_id"],
    max_encoder_length=48,   # 4 ساعت تاریخچه
    max_prediction_length=6,  # 30 دقیقه prediction
    time_varying_known_reals=["hour", "day_of_week", "temperature"],
    time_varying_unknown_reals=["demand"],
)

model = TemporalFusionTransformer.from_dataset(dataset)
```

**۴. Graph Neural Network — برای Spatial Correlation:**

سلول‌ها به هم مرتبط هستند. تقاضا در یک منطقه بر همسایگان تأثیر می‌گذارد:

```python
import torch_geometric

class DemandGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GATConv(in_channels=feature_dim, out_channels=64)
        self.lstm = nn.LSTM(64, 32)
        self.output = nn.Linear(32, 1)
    
    def forward(self, x, edge_index, edge_weight):
        # Spatial aggregation
        spatial = self.gnn(x, edge_index, edge_weight)
        # Temporal modeling
        temporal, _ = self.lstm(spatial)
        return self.output(temporal)
```

**Uncertainty Quantification:**

پیش‌بینی یک عدد کافی نیست. باید uncertainty هم اعلام کنیم:
- **Quantile Regression:** پیش‌بینی P10, P50, P90
- **Conformal Prediction:** prediction intervals با coverage guarantee

```python
# Quantile Loss
def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return torch.mean(torch.maximum(quantile * error, (quantile - 1) * error))
```

**Pipeline کامل:**

```
Raw Events (Kafka) → Feature Pipeline (Flink) → Feature Store (Redis)
                                                        ↓
Model Training (Weekly) → Model Registry → Inference Engine → Dispatch System
                                                        ↓
                                              Monitoring Dashboard
                                           (MAE, MAPE, Bias per cell)
```

**Evaluation:**

- **MAPE:** Mean Absolute Percentage Error — مناسب برای مقایسه مناطق مختلف
- **MAE:** برای error در واحد تعداد
- **Winkler Score:** برای ارزیابی prediction intervals
- **Business Metric:** driver positioning efficiency, ETA accuracy

**مثال واقعی:**

Uber از یک hierarchical forecasting approach استفاده می‌کند:
- سطح شهر → سطح منطقه → سطح cell
- Bottom-up reconciliation برای consistency
- Ensemble از چند مدل

**اشتباهات رایج کاندیداها:**
- Spatial correlation را نادیده می‌گیرند
- Cold start برای مناطق جدید را فراموش می‌کنند
- Uncertainty را در پیش‌بینی در نظر نمی‌گیرند

**سوالات Follow-up:**
- چطور با extreme events (توفان، جشن سال نو) کنار می‌آیید؟
- Hierarchical forecasting چیست و چه مزایایی دارد؟
- چطور causal inference را برای تخمین تأثیر promotions استفاده می‌کنید؟

---

### سوال ۱۵: طراحی Large-Scale Language Model Training Infrastructure

**دسته:** DL System Design / Distributed Training  
**سطح:** Staff/Principal  
**چرا مهمه:** با ظهور LLMها، توانایی طراحی infrastructure آموزش مدل‌های بزرگ به یک مهارت کلیدی برای senior AI engineers تبدیل شده است. این سوال عمیق‌ترین سطح system design را می‌سنجد.  
**زمان پاسخ:** ۳۰ تا ۵۰ دقیقه

**سوال:** Infrastructure آموزش یک Language Model با ۷۰ میلیارد پارامتر (مشابه Llama-3 70B) روی ۱۰۲۴ GPU طراحی کنید. چه استراتژی‌های parallelism استفاده می‌کنید؟ چطور با node failure و memory constraints کنار می‌آیید؟

**پاسخ کامل:**

**چالش‌های اصلی:**

یک مدل ۷۰B با:
- وزن‌ها: ۷۰B × 2 bytes (BF16) = ~۱۴۰ GB
- Optimizer states (Adam): × 3 = ~۴۲۰ GB فقط برای آموزش
- Activations در طول forward pass: چندین برابر بیشتر

یک A100 80GB GPU → حتی وزن‌های مدل جا نمی‌شود. باید distributed training داشت.

**استراتژی‌های Parallelism:**

**۱. Data Parallelism (DP):**

همه GPUها یک کپی کامل از مدل دارند. هر GPU روی یک mini-batch متفاوت کار می‌کند. گرادیان‌ها با AllReduce sync می‌شوند.

محدودیت: مدل باید در یک GPU جا شود.

**ZeRO (Zero Redundancy Optimizer):**

DeepSpeed ZeRO در سه مرحله redundancy را حذف می‌کند:
- **ZeRO-1:** Optimizer states را بین GPUها shard می‌کند (4x کاهش memory)
- **ZeRO-2:** Gradients هم shard می‌شوند (8x کاهش)
- **ZeRO-3:** Model parameters هم shard می‌شوند (کامل‌ترین صرفه‌جویی)

```python
import deepspeed

ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # ZeRO-Infinity
        "offload_param": {"device": "cpu"}
    },
    "bf16": {"enabled": True},
    "train_micro_batch_size_per_gpu": 4
}

model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

**۲. Tensor Parallelism (TP):**

هر لایه به صورت horizontal بین GPUها تقسیم می‌شود. برای Transformer:
- قبل از attention: هر head روی یک GPU
- Linear layers: columns یا rows تقسیم می‌شوند

```
GPU 0: Attention heads 1-8    |  Linear: cols 1-2048
GPU 1: Attention heads 9-16   |  Linear: cols 2049-4096
```

Communication overhead: AllReduce در هر لایه. مناسب برای GPUها روی یک node (NVLink سریع‌تر از inter-node).

**۳. Pipeline Parallelism (PP):**

مدل به صورت عمودی تقسیم می‌شود. هر GPU مسئول چند لایه است.

```
GPU 0: Layers 1-20   →   GPU 1: Layers 21-40   →   GPU 2: Layers 41-60   →   GPU 3: Layers 61-80
```

**Micro-batching:** برای کاهش idle time (pipeline bubble):
```
t=1: GPU0 forward micro-batch 1
t=2: GPU0 forward micro-batch 2 | GPU1 forward micro-batch 1
t=3: GPU0 forward micro-batch 3 | GPU1 forward micro-batch 2 | GPU2 forward micro-batch 1
...
```

**۳D Parallelism (Megatron-LM):**

ترکیب DP + TP + PP:
```
۱۰۲۴ GPU:
- ۴ way TP (یک node)
- ۸ way PP (بین nodes)
- ۳۲ way DP
= 4 × 8 × 32 = 1024 GPU
```

**Memory Optimization:**

**Gradient Checkpointing:**

به جای نگه داشتن همه activations در حین forward، فقط نقاط checkpoint نگه داشته می‌شوند. Activations در حین backward دوباره محاسبه می‌شوند.

Trade-off: ۳۰-۴۰٪ بیشتر compute اما ۳-۴x کاهش activation memory.

**Mixed Precision Training:**

```python
# BF16 vs FP16:
# FP16: dynamic range محدود → gradient underflow
# BF16: همان dynamic range FP32 با نصف حافظه
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Master weights در FP32 نگه می‌شوند برای stability
```

**Flash Attention:**

الگوریتم attention بازنویسی شده که:
- IO-aware: داده‌ها بین HBM و SRAM GPU بهینه منتقل می‌شوند
- ۲-۴x سریع‌تر از standard attention
- ذاکره خطی به جای quadratic برای sequence طولانی

**Fault Tolerance:**

با ۱۰۲۴ GPU، احتمال خرابی node در طول training هفته‌ها بالاست.

**Checkpointing Strategy:**
```python
# هر ۵۰۰ step ذخیره‌سازی
if global_step % 500 == 0:
    save_checkpoint({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"checkpoint_step_{global_step}.pt")
```

**Elastic Training:** با PyTorch Elastic، وقتی یک node خراب شود، training با تعداد کمتری GPU ادامه می‌یابد.

**Gradient Clipping:** جلوگیری از gradient explosion:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Monitoring در طول Training:**

- Loss curve (باید smooth باشد و به تدریج کاهش یابد)
- Gradient norm (نباید خیلی بزرگ یا خیلی کوچک باشد)
- GPU utilization (باید > 90٪ باشد)
- MFU (Model FLOPs Utilization): عملکرد واقعی vs نظری GPU
- Throughput: tokens/second

**مثال واقعی — Meta's Training Setup:**

Llama-3 70B روی ~۱۶K H100 GPU آموزش دید با:
- Megatron-LM برای 3D parallelism
- Flash Attention 2
- Checkpoint هر ساعت به storage پایدار
- Custom networking با InfiniBand و RoCE برای inter-node communication

**اشتباهات رایج کاندیداها:**
- فکر می‌کنند data parallelism برای مدل‌های بزرگ کافی است
- Pipeline bubble را در نظر نمی‌گیرند
- Fault recovery strategy ندارند

**سوالات Follow-up:**
- تفاوت Megatron-LM و DeepSpeed چیست؟
- Ring AllReduce چطور کار می‌کند؟
- چطور MFU را اندازه می‌گیرید و بهبود می‌دهید؟

---

*پایان سوالات ۲۰۲۶-۰۴-۲۱ | نوشته‌شده توسط FAANG-level Interview Coach*
