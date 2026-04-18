# سوالات مصاحبه فنی — ۲۰۲۶-۰۴-۱۸

**دسته‌بندی‌ها:** Data Science | AI Engineering | DL/ML System Design
**سطح هدف:** Mid / Senior / Staff Engineer

---

## بخش اول: Data Science

---
### سوال ۱: مدیریت عدم تعادل کلاس‌ها در داده‌های آموزشی
**دسته:** Classification / Data Preprocessing
**سطح:** Mid/Senior
**چرا مهمه:** در اکثر مسائل دنیای واقعی (تشخیص تقلب، تشخیص بیماری، تشخیص نقص) داده‌ها به‌شدت نامتعادل هستند و مدل‌های ساده به سمت کلاس اکثریت bias می‌شوند.
**زمان پاسخ:** ۱۵ تا ۲۵ دقیقه

**سوال:** فرض کنید دارید یک مدل تشخیص تقلب (Fraud Detection) می‌سازید و داده‌های شما ۹۹٪ تراکنش سالم و ۱٪ تراکنش تقلبی دارد. چگونه با این عدم تعادل کنار می‌آیید؟ trade-off‌های هر رویکرد را توضیح دهید.

**پاسخ کامل:**

عدم تعادل کلاس‌ها (Class Imbalance) یکی از چالش‌برانگیزترین مشکلات عملی در Machine Learning است. وقتی مدلی روی داده‌های نامتعادل آموزش می‌بیند، ممکن است Accuracy بالایی نشان دهد اما در واقع کاملاً بی‌فایده باشد — برای مثال یک مدل که همیشه "سالم" پیش‌بینی کند، Accuracy 99% دارد اما F1-Score روی کلاس اقلیت صفر است.

**رویکرد ۱: تغییر متریک ارزیابی**

اولین قدم نه تغییر داده، بلکه تغییر نحهٔ اندازه‌گیری عملکرد است. به جای Accuracy باید از:
- **Precision-Recall AUC**: وقتی اهمیت False Negative بالاست (مثل miss کردن تقلب)
- **F1-Score یا Fβ-Score**: اگر β > 1 باشد، Recall بیشتر وزن می‌گیرد
- **ROC-AUC**: مناسب برای مقایسه کلی مدل‌ها
- **Matthews Correlation Coefficient (MCC)**: متریک بسیار قوی برای داده نامتعادل که همه چهار مقدار TP, TN, FP, FN را در نظر می‌گیرد

**رویکرد ۲: Resampling — سطح داده**

دو استراتژی اصلی وجود دارد:

*Oversampling* — افزایش نمونه‌های کلاس اقلیت:
- **Random Oversampling**: کپی تصادفی نمونه‌های minority. ساده اما باعث overfitting می‌شود چون مدل همان نمونه‌ها را بارها می‌بیند.
- **SMOTE (Synthetic Minority Oversampling Technique)**: نمونه‌های مصنوعی بین نمونه‌های موجود در فضای feature ایجاد می‌کند. با interpolation بین K نزدیک‌ترین همسایه کار می‌کند. بهتر از random oversampling است اما ممکن است نمونه‌های overlapping ایجاد کند.
- **ADASYN (Adaptive Synthetic Sampling)**: مثل SMOTE اما بیشتر روی نمونه‌های دشوار تمرکز می‌کند.

*Undersampling* — کاهش نمونه‌های کلاس اکثریت:
- **Random Undersampling**: حذف تصادفی. اطلاعات ارزشمند از دست می‌رود.
- **Tomek Links**: جفت‌هایی که از دو کلاس مختلف هستند و نزدیک‌ترین همسایه یکدیگرند حذف می‌شوند. مرز تصمیم را تمیزتر می‌کند.
- **NearMiss**: نمونه‌های majority را بر اساس فاصله از minority انتخاب می‌کند.

ترکیب SMOTE + Tomek Links یا SMOTE + ENN (Edited Nearest Neighbors) معمولاً بهترین نتایج را می‌دهد.

**رویکرد ۳: تغییر وزن کلاس — سطح الگوریتم**

اکثر مدل‌های sklearn پارامتر `class_weight='balanced'` دارند. این کار باعث می‌شود loss function برای خطاهای کلاس اقلیت جریمه بیشتری در نظر بگیرد. فرمول وزن:

```
w_i = n_samples / (n_classes × n_samples_class_i)
```

این رویکرد ساده‌ترین و اغلب موثرترین راه است و نیازی به تغییر داده ندارد.

**رویکرد ۴: الگوریتم‌های خاص**

- **Ensemble Methods مخصوص Imbalance**: مثل BalancedRandomForest، EasyEnsemble، RUSBoost
- **One-Class Classification**: مدل فقط روی کلاس اکثریت آموزش می‌بیند و هر چیز غیرعادی را تقلب تلقی می‌کند (مثل Isolation Forest یا Autoencoder)
- **Cost-Sensitive Learning**: مستقیماً cost matrix تعریف می‌کنیم — miss کردن یک تقلب چقدر گران‌تر از false alarm است؟

**رویکرد ۵: Threshold Tuning**

به جای پیش‌فرض 0.5 برای classification threshold، می‌توان آستانه را بهینه کرد. با رسم Precision-Recall Curve می‌توان threshold را بر اساس نیاز کسب‌وکار تنظیم کرد — مثلاً در تشخیص تقلب ترجیح می‌دهیم Recall بالا داشته باشیم حتی به قیمت Precision پایین‌تر.

**اشتباهات رایج کاندیداها:**
- اعمال SMOTE روی کل داده قبل از تقسیم train/test — این باعث data leakage می‌شود. Resampling باید فقط روی training set انجام شود.
- فراموش کردن که test set باید distribution واقعی را منعکس کند.
- استفاده از Accuracy به عنوان متریک اصلی.
- اعمال oversampling بدون بررسی اینکه آیا overlap بین کلاس‌ها در فضای feature وجود دارد یا نه.

**سوالات follow-up احتمالی:**
- "اگر داده جدید در production نیز همان distribution را داشته باشد، آیا resampling واقعاً کمک می‌کند؟"
- "چطور می‌فهمید که مدل شما به اندازه کافی Recall دارد؟ این threshold را چطور با business تعیین می‌کنید؟"
- "در سیستم real-time fraud detection، چه latency constraint هایی دارید و چطور با آن‌ها کنار می‌آیید؟"

---
### سوال ۲: Feature Engineering برای داده‌های سری زمانی
**دسته:** Feature Engineering / Time Series
**سطح:** Senior
**چرا مهمه:** اکثر داده‌های دنیای واقعی ماهیت زمانی دارند — از قیمت سهام تا رفتار کاربر و سیگنال‌های IoT. Feature Engineering مناسب اغلب مهم‌تر از انتخاب مدل است.
**زمان پاسخ:** ۲۰ تا ۳۰ دقیقه

**سوال:** برای یک مسئله پیش‌بینی (forecasting) فروش روزانه یک محصول، چه feature هایی از داده‌های تاریخی استخراج می‌کنید؟ چه اشتباهاتی ممکن است باعث data leakage شوند؟

**پاسخ کامل:**

Feature Engineering برای سری‌های زمانی یک هنر ترکیب دانش دامنه‌ای (domain knowledge) با تکنیک‌های آماری است. ساختار این feature ها معمولاً در چند دسته کلی قرار می‌گیرند:

**دسته ۱: Feature های تقویمی (Calendar Features)**

ساده‌ترین و اغلب موثرترین feature ها، ویژگی‌های مبتنی بر زمان هستند:
- **روز هفته** (day_of_week): فروش معمولاً در روزهای مختلف هفته الگوی متفاوتی دارد.
- **ماه و فصل**: رفتار فصلی در اکثر کسب‌وکارها وجود دارد.
- **روز ماه**: برخی محصولات در ابتدا یا انتهای ماه (زمان دریافت حقوق) پیک فروش دارند.
- **تعطیلات رسمی**: با یک تقویم تعطیلات، flag هایی مثل `is_holiday`, `days_to_holiday`, `days_after_holiday` بسازید.
- **ویژگی‌های چرخه‌ای (Cyclical Encoding)**: روز هفته را نباید به عنوان عدد خطی (0 تا 6) نشان داد زیرا مدل متوجه نمی‌شود که شنبه و یکشنبه نزدیک هم هستند. به جای آن از sin/cos encoding استفاده کنید:
  ```
  day_sin = sin(2π × day_of_week / 7)
  day_cos = cos(2π × day_of_week / 7)
  ```

**دسته ۲: Lag Features (ویژگی‌های تاخیری)**

مقادیر گذشته متغیر هدف به عنوان feature:
- `lag_1`: فروش دیروز
- `lag_7`: فروش همین روز هفته گذشته
- `lag_14`, `lag_28`: الگوهای دو و چهار هفته پیش
- `lag_365`: فروش همین روز سال گذشته (seasonal lag)

**دسته ۳: Rolling Statistics (آمار پنجره‌ای)**

- `rolling_mean_7`: میانگین فروش ۷ روز گذشته — نشانه روند کوتاه‌مدت
- `rolling_std_7`: انحراف معیار ۷ روز — نشانه نوسان
- `rolling_max_30`: حداکثر فروش ۳۰ روز گذشته
- `ewm_7` (Exponentially Weighted Mean): میانگین وزن‌دار که به داده‌های اخیر وزن بیشتری می‌دهد

**دسته ۴: Decomposition Features**

با استفاده از STL (Seasonal and Trend decomposition using Loess) یا مشابه، سری زمانی را به سه بخش تجزیه کنید:
- **Trend**: روند بلندمدت
- **Seasonality**: الگوی فصلی
- **Residual**: باقی‌مانده تصادفی

این سه بخش می‌توانند به عنوان feature استفاده شوند یا به صورت جداگانه مدل‌سازی شوند.

**دسته ۵: Feature های خارجی (Exogenous Variables)**

- قیمت رقبا (اگر در دسترس باشد)
- شاخص‌های اقتصادی مرتبط
- داده‌های آب‌وهوایی برای محصولات وابسته به فصل
- کمپین‌های تبلیغاتی
- رویدادهای ویژه (مسابقات ورزشی، انتخابات)

**خطر بزرگ: Data Leakage در سری زمانی**

این مهم‌ترین بخش پاسخ است. Data Leakage در سری‌های زمانی یعنی استفاده از اطلاعات آینده برای پیش‌بینی گذشته، که در production کاملاً غیرممکن است:

۱. **اشتباه در تقسیم train/test**: هرگز از `random_split` استفاده نکنید. باید داده‌ها به صورت زمانی تقسیم شوند — همیشه گذشته برای آموزش، آینده برای test.

۲. **محاسبه rolling stats از آینده**: اگر برای row تاریخ ۱۵ام ماه، `rolling_mean_7` را با داده‌های ۱۵ تا ۲۱ام حساب کنید، leaked است. همیشه باید از داده‌های قبل از آن روز محاسبه شود.

۳. **Target encoding بدون cross-validation**: اگر target encoding را روی کل داده حساب کنید و بعد تقسیم کنید، اطلاعات آینده leak شده.

۴. **Normalization با آمار کل**: اگر min-max scaling را با min و max کل dataset (شامل test) حساب کنید، leaked است. باید فقط از training set حساب کنید.

```python
# اشتباه — leakage
df['rolling_mean'] = df['sales'].rolling(7).mean()

# درست — shift به اندازه horizon پیش‌بینی
df['rolling_mean'] = df['sales'].shift(1).rolling(7).mean()
```

**اشتباهات رایج کاندیداها:**
- فراموش کردن Cyclical Encoding برای ویژگی‌های زمانی.
- نادیده گرفتن تاثیر تعطیلات و رویدادهای ویژه.
- استفاده از lag هایی که در horizon پیش‌بینی در دسترس نخواهند بود.
- عدم بررسی Autocorrelation (ACF) و Partial Autocorrelation (PACF) برای یافتن lag های مناسب.

**سوالات follow-up احتمالی:**
- "چطور با missing values در سری زمانی کنار می‌آیید؟"
- "اگر فروش یک محصول جدید را پیش‌بینی کنید که تاریخچه کافی ندارد، چطور عمل می‌کنید؟ (Cold Start Problem)"
- "چه معماری مدلی — ARIMA، Prophet، XGBoost، یا LSTM — برای این مسئله انتخاب می‌کنید و چرا؟"

---
### سوال ۳: آزمون فرض آماری — A/B Testing در محیط Production
**دسته:** Statistics / Experimentation
**سطح:** Senior/Staff
**چرا مهمه:** هر تغییر در product نیاز به تصمیم‌گیری مبتنی بر داده دارد. درک صحیح A/B Testing مرز بین تصمیمات درست و اشتباه است.
**زمان پاسخ:** ۲۰ تا ۳۰ دقیقه

**سوال:** تیم شما می‌خواهد بفهمد آیا یک دکمه جدید روی صفحه پرداخت، نرخ تبدیل (Conversion Rate) را افزایش می‌دهد. چطور یک A/B test طراحی می‌کنید؟ چه اشتباهات آماری باید از آن‌ها اجتناب کنید؟

**پاسخ کامل:**

A/B Testing یا Controlled Experiment یکی از قوی‌ترین ابزارهای علّی (causal inference) در دسترس ما است. برخلاف تحلیل‌های observational، وقتی درست انجام شود می‌توانیم با اطمینان بگوییم یک تغییر چه اثری داشته.

**مرحله ۱: تعریف دقیق فرضیه و متریک**

قبل از هر چیز باید سوال را دقیق تعریف کنیم:
- **Primary Metric**: نرخ تبدیل (Conversion Rate) — نسبت کاربرانی که خرید می‌کنند به کل بازدیدکنندگان
- **Secondary Metrics (Guardrail Metrics)**: متریک‌هایی که نباید بد شوند — مثل میانگین ارزش سفارش، نرخ لغو، زمان صرف شده در صفحه
- **فرضیه صفر (H₀)**: تفاوتی بین Control و Treatment نیست (CR_A = CR_B)
- **فرضیه جایگزین (H₁)**: CR_B ≠ CR_A (two-tailed) یا CR_B > CR_A (one-tailed)

**مرحله ۲: محاسبه Sample Size (قبل از شروع)**

این مهم‌ترین قدم است که اکثراً نادیده گرفته می‌شود. باید قبل از شروع آزمایش بدانیم چقدر داده نیاز داریم:

```
n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²
```

که در آن:
- **α (Significance Level)**: احتمال خطای نوع اول (False Positive) — معمولاً 0.05
- **β**: احتمال خطای نوع دوم (False Negative) — معمولاً 0.20
- **Statistical Power (1-β)**: توانایی تشخیص اثر واقعی — معمولاً 80% یا 90%
- **Minimum Detectable Effect (MDE)**: کوچک‌ترین تفاوتی که اهمیت کسب‌وکاری دارد — مثلاً 2% بهبود در CR

اگر baseline CR = 5% باشد و MDE = 1% (می‌خواهیم 5% به 6% برود)، با α=0.05 و power=80% به حدود ۱۴,۰۰۰ کاربر در هر گروه نیاز داریم.

**مرحله ۳: تخصیص تصادفی**

- **Unit of Randomization**: کاربر (User ID)، session، یا device؟ باید unit انتخاب شده با unit آنالیز یکی باشد.
- **Stratification**: برای کاهش واریانس، می‌توان بر اساس ویژگی‌های مهم (مثل platform، کشور) stratify کرد.
- **Salt در Hashing**: `hash(user_id + experiment_id)` تضمین می‌کند هر کاربر همیشه به همان گروه برود (Sticky Assignment).

**مرحله ۴: اجرا و مانیتورینگ**

- **SRM Check (Sample Ratio Mismatch)**: آیا نسبت کاربران در دو گروه واقعاً 50-50 است؟ اگر نه، مشکل در randomization داریم.
- **هرگز در وسط تست متوقف نشوید**: این مهم‌ترین اشتباه در A/B Testing است.

**اشتباه بزرگ ۱: Peeking Problem (آمار همیشه نگاه کردن)**

اگر هر روز p-value را چک کنید و به محض رسیدن به p < 0.05 متوقف شوید، خطای نوع اول واقعی شما بسیار بالاتر از 5% است. این به "Multiple Testing Problem" یا "Optional Stopping" معروف است.

راه‌حل‌ها:
- **Sequential Testing**: از روش‌هایی مثل SPRT (Sequential Probability Ratio Test) یا Alpha Spending Functions استفاده کنید که برای interim analysis طراحی شده‌اند.
- **Bayesian A/B Testing**: به جای p-value، از Probability of Being Best یا Expected Loss استفاده کنید.
- **یا ساده‌ترین راه**: sample size را از قبل محاسبه کنید و صبر کنید.

**اشتباه بزرگ ۲: Network Effects و Interference**

اگر کاربران با هم تعامل دارند (مثل شبکه‌های اجتماعی)، randomization در سطح کاربر کافی نیست. کاربر Treatment ممکن است روی کاربر Control تاثیر بگذارد. راه‌حل: Cluster Randomization (تخصیص بر اساس گروه‌های مستقل).

**اشتباه بزرگ ۳: Novelty Effect**

اگر کاربران فقط به خاطر جدید بودن یک feature با آن تعامل می‌کنند، نتایج کوتاه‌مدت گمراه‌کننده است. باید آزمایش را به اندازه کافی طولانی نگه دارید.

**تحلیل نهایی:**

پس از جمع‌آوری داده کافی، آزمون دو نسبت (Two-proportion Z-test) انجام می‌دهیم:

```
z = (p̂₁ - p̂₂) / sqrt(p̂(1-p̂)(1/n₁ + 1/n₂))
```

که p̂ میانگین وزن‌دار دو نسبت است. اگر |z| > z_{α/2} ≈ 1.96، H₀ را رد می‌کنیم.

**اشتباهات رایج کاندیداها:**
- گفتن "اگر p < 0.05 باشد، تست موفق است" بدون توجه به effect size و significance کسب‌وکاری.
- فراموش کردن guardrail metrics — ممکن است CR بالا رود ولی میانگین ارزش سفارش پایین بیاید.
- نادیده گرفتن carryover effects (اثر آزمایش‌های قبلی).

**سوالات follow-up احتمالی:**
- "اگر بخواهید بیش از دو variant را با هم مقایسه کنید، چه تغییری در آزمون نیاز است؟ (Multiple Comparison Correction)"
- "Bayesian vs Frequentist A/B Testing — چه زمانی کدام را ترجیح می‌دهید؟"
- "اگر ترافیک سایت خیلی کم باشد و رسیدن به sample size ماه‌ها طول بکشد، چه می‌کنید؟"

---
### سوال ۴: Causal Inference — وقتی A/B Test ممکن نیست
**دسته:** Causal Inference / Statistics
**سطح:** Staff/Principal
**چرا مهمه:** در دنیای واقعی اغلب نمی‌توان آزمایش کنترل‌شده انجام داد. توانایی استنتاج علّی از داده‌های observational یک مهارت Staff-level است.
**زمان پاسخ:** ۲۵ تا ۳۵ دقیقه

**سوال:** شرکت شما یک برنامه آموزشی برای کارمندان اجرا کرده، اما شرکت در آن اجباری نبوده. حالا می‌خواهید بفهمید آیا این برنامه واقعاً بر بهره‌وری تاثیر داشته. با داده‌های observational (غیرآزمایشی) چطور این مسئله را حل می‌کنید؟

**پاسخ کامل:**

این مسئله کلاسیک Confounding در causal inference است. کسانی که در برنامه شرکت کرده‌اند ممکن است از ابتدا motivated‌تر بوده باشند و اگر صرفاً میانگین بهره‌وری دو گروه را مقایسه کنیم، این تفاوت را به برنامه نسبت می‌دهیم نه به selection bias.

**چارچوب نظری: Potential Outcomes (Rubin Causal Model)**

برای هر فرد، دو potential outcome داریم:
- Y(1): بهره‌وری اگر در برنامه شرکت کرده بود
- Y(0): بهره‌وری اگر شرکت نکرده بود

**Fundamental Problem of Causal Inference**: برای هر فرد، فقط یکی از این دو را مشاهده می‌کنیم. **Average Treatment Effect (ATE)** عبارت است از E[Y(1) - Y(0)].

**روش ۱: Matching**

برای هر نفری که شرکت کرده، یک نفر از گروه کنترل پیدا می‌کنیم که از نظر covariates مشابه است. Matching می‌تواند بر اساس:
- **Exact Matching**: بر اساس تمام ویژگی‌ها — در ابعاد بالا ناممکن می‌شود (Curse of Dimensionality)
- **Propensity Score Matching (PSM)**: ابتدا احتمال شرکت در برنامه را بر اساس covariates مدل‌سازی می‌کنیم (با Logistic Regression)، سپس بر اساس این propensity score matching انجام می‌دهیم. این کار ابعاد را به یک عدد کاهش می‌دهد.
- **Mahalanobis Distance Matching**: فاصله بر اساس covariance matrix داده‌ها

**روش ۲: Difference-in-Differences (DiD)**

اگر داده‌های قبل و بعد از برنامه داشته باشیم:

```
ATE = (Y_treatment_after - Y_treatment_before) - (Y_control_after - Y_control_before)
```

این روش فرض می‌کند روند تغییر در دو گروه در غیاب treatment یکسان بود (**Parallel Trends Assumption**). باید این assumption را با داده‌های قبل از treatment بررسی کرد.

**روش ۳: Instrumental Variables (IV)**

اگر متغیری وجود داشته باشد که:
۱. بر شرکت در برنامه تاثیر دارد (relevance)
۲. فقط از طریق شرکت در برنامه بر بهره‌وری تاثیر دارد، نه مستقیم (exclusion restriction)
۳. با confounders ارتباط ندارد (exogeneity)

مثال: اگر برنامه در یک شهر خاص اجرا شد، فاصله از آن شهر می‌تواند instrument باشد. Two-Stage Least Squares (2SLS) برای برآورد اثر استفاده می‌شود. این روش LATE (Local Average Treatment Effect) را برآورد می‌کند — اثر برای کسانی که به خاطر instrument تصمیمشان تغییر کرد.

**روش ۴: Regression Discontinuity Design (RDD)**

اگر دسترسی به برنامه بر اساس یک آستانه (threshold) تعیین شده — مثلاً فقط کارمندانی با سابقه زیر ۲ سال می‌توانستند شرکت کنند — می‌توان افراد درست اطراف این آستانه را مقایسه کرد چون در آن نقطه، تخصیص شبه‌تصادفی است.

**روش ۵: Doubly Robust Estimation**

ترکیب Propensity Score و مدل Outcome به گونه‌ای که اگر هر یک درست باشد، برآورد درست است:

```
ATE = E[(Y × D/e(X)) - (Y × (1-D)/(1-e(X))) - (μ₁(X) - μ₀(X))]
```

که e(X) propensity score و μ₁, μ₀ مدل‌های پیش‌بینی outcome هستند.

**ابزارهای عملی:**
- `dowhy` در Python: چارچوب کامل برای causal inference
- `econml` از Microsoft: برای Heterogeneous Treatment Effects
- `causalml` از Uber: برای uplift modeling

**اشتباهات رایج کاندیداها:**
- استفاده از correlation به عنوان causal inference بدون توجه به confounders.
- فراموش کردن بررسی assumptions (مثل Parallel Trends در DiD).
- گفتن "داریم کنترل می‌کنیم برای X" بدون توجه به Collider Bias — کنترل کردن برای برخی متغیرها می‌تواند bias جدید ایجاد کند.

**سوالات follow-up احتمالی:**
- "چطور وجود unobserved confounders را بررسی می‌کنید؟ (Sensitivity Analysis)"
- "اگر اثر برنامه برای افراد مختلف متفاوت باشد (Heterogeneous Treatment Effects)، چطور آن را مدل‌سازی می‌کنید؟"
- "فرق بین ATE, ATT, ATU و LATE را توضیح دهید."

---
### سوال ۵: Cross-Validation — انتخاب درست استراتژی
**دسته:** Model Evaluation / Validation
**سطح:** Mid/Senior
**چرا مهمه:** انتخاب اشتباه cross-validation می‌تواند منجر به ارزیابی کاذب خوش‌بینانه شود که در production شکست می‌خورد.
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** چه زمانی از K-Fold Cross-Validation استفاده می‌کنید و چه زمانی به استراتژی دیگری نیاز دارید؟ مثال‌های واقعی بزنید.

**پاسخ کامل:**

Cross-Validation به ما کمک می‌کند تخمین بهتری از عملکرد واقعی مدل در داده‌های ندیده داشته باشیم. اما یک استراتژی برای همه مسائل مناسب نیست.

**K-Fold Cross-Validation — استاندارد**

داده را به K قسمت مساوی تقسیم می‌کنیم. در هر fold، یک قسمت test و بقیه train هستند. K=5 یا K=10 رایج است.

*مناسب برای:* مسائل classification و regression با داده‌های iid (Independent and Identically Distributed). مثلاً پیش‌بینی قیمت خانه بر اساس ویژگی‌های خانه.

*محدودیت‌ها:*
- برای داده‌های سری زمانی ناصحیح است (leakage از آینده به گذشته)
- برای داده‌های گروه‌بندی‌شده نامناسب است

**Stratified K-Fold**

مثل K-Fold اما نسبت کلاس‌ها در هر fold حفظ می‌شود. **برای classification مسائل نامتعادل الزامی است.** اگر dataset شما 5% کلاس مثبت دارد، ممکن است یک fold هیچ نمونه مثبتی نداشته باشد.

**Time Series Cross-Validation (Walk-Forward Validation)**

برای داده‌های سری زمانی باید همیشه از گذشته برای پیش‌بینی آینده استفاده کرد:

```
Fold 1: Train=[1..100], Test=[101..110]
Fold 2: Train=[1..110], Test=[111..120]
Fold 3: Train=[1..120], Test=[121..130]
```

در هر fold، training set بزرگ‌تر می‌شود. این روش **TimeSeriesSplit** در sklearn نامیده می‌شود. برخی پیاده‌سازی‌ها اجازه می‌دهند gap بین train و test باشد تا اثر leakage از نزدیک‌ترین داده‌ها حذف شود.

**Group K-Fold**

وقتی داده‌ها گروه‌بندی دارند و نمونه‌های یک گروه نباید در هر دو train و test باشند. مثال‌ها:
- پیش‌بینی بیماری: اگر از یک بیمار چند نمونه دارید، همه نمونه‌های یک بیمار باید در یک fold باشند. در غیر این صورت مدل به جای یادگیری الگوی کلی، بیماران خاص را "حفظ" می‌کند.
- تشخیص تصویر: اگر از یک دوربین چند تصویر دارید، باید همه در یک fold باشند.
- پیش‌بینی بر اساس کاربر: نمونه‌های یک کاربر نباید در هر دو train و test باشند.

**Leave-One-Out Cross-Validation (LOOCV)**

K = N (هر نمونه یک fold جداگانه). برآورد بسیار دقیق‌تری می‌دهد اما N برابر گران‌تر از یک train است. تنها برای dataset های خیلی کوچک (N < 50) معقول است.

**Nested Cross-Validation**

برای hyperparameter tuning استفاده می‌شود تا از evaluation leakage جلوگیری شود:
- **حلقه بیرونی**: ارزیابی عملکرد مدل نهایی
- **حلقه داخلی**: انتخاب hyperparameter های بهینه

اگر از همان داده‌ای که برای hyperparameter tuning استفاده کرده‌اید برای evaluation استفاده کنید، برآورد شما بیش از حد خوش‌بینانه است.

**Adversarial Validation — تکنیک پیشرفته**

گاهی اوقات distribution داده‌های train و test متفاوت است (Distribution Shift). می‌توان یک classifier ساخت که تلاش کند train و test را از هم تشخیص دهد. اگر AUC این classifier بالا باشد، یعنی train و test از هم متفاوت هستند و باید نگران generalization باشیم.

**جدول راهنما:**

| سناریو | Cross-Validation مناسب |
|--------|------------------------|
| داده iid، balanced | K-Fold |
| داده iid، imbalanced | Stratified K-Fold |
| سری زمانی | Time Series Split |
| داده گروه‌بندی شده | Group K-Fold |
| dataset خیلی کوچک | LOOCV |
| hyperparameter tuning | Nested CV |

**اشتباهات رایج کاندیداها:**
- اعمال K-Fold روی سری زمانی و ایجاد data leakage از آینده.
- فراموش کردن که preprocessing (مثل standardization یا feature selection) باید داخل هر fold انجام شود، نه قبل از split.
- گزارش دادن عملکرد بر اساس validation set که برای انتخاب مدل استفاده شده — باید یک test set کاملاً ندیده داشته باشید.

**سوالات follow-up احتمالی:**
- "Bias-Variance Tradeoff در K-Fold چطور است؟ K بزرگ‌تر چه تاثیری دارد؟"
- "چطور با Pipeline در sklearn مطمئن می‌شوید preprocessing داخل fold اتفاق می‌افتد؟"
- "اگر training مدل خیلی گران باشد، چطور cross-validation را بهینه می‌کنید؟"

---

## بخش دوم: AI Engineering

---
### سوال ۶: طراحی سیستم RAG (Retrieval-Augmented Generation)
**دسته:** LLM Systems / Architecture
**سطح:** Senior/Staff
**چرا مهمه:** RAG یکی از رایج‌ترین معماری‌های LLM در production است و درک عمیق trade-off های آن برای هر AI Engineer ضروری است.
**زمان پاسخ:** ۲۵ تا ۳۵ دقیقه

**سوال:** می‌خواهید یک سیستم Q&A بسازید که بتواند سوالات کاربران را بر اساس مستندات داخلی شرکت (۱۰۰,۰۰۰+ سند) پاسخ دهد. معماری RAG خود را طراحی کنید و trade-off های هر تصمیم را توضیح دهید.

**پاسخ کامل:**

RAG (Retrieval-Augmented Generation) یک معماری است که محدودیت‌های LLM های pure generative را با یک مرحله retrieval از دانش خارجی برطرف می‌کند. این معماری به مشکل hallucination، outdated knowledge و thinh context window رسیدگی می‌کند.

**معماری کلی:**

```
[User Query] → [Query Processing] → [Retrieval Engine] → [Re-ranking] → [Context Assembly] → [LLM Generation] → [Response]
         ↑                                    ↑
    [Query Expansion]              [Vector Database + BM25]
```

**مرحله ۱: Indexing Pipeline (آفلاین)**

*Document Processing:*
- **Parsing**: استخراج متن از PDF، Word، HTML با ابزارهایی مثل PyMuPDF، Unstructured.io
- **Chunking Strategy**: یکی از مهم‌ترین تصمیمات. گزینه‌ها:
  - Fixed-size chunks با overlap (مثلاً ۵۱۲ token با ۵۰ token overlap): ساده اما ممکن است مفهوم را قطع کند
  - Semantic chunking: تقسیم بر اساس مرز جملات و پاراگراف‌ها
  - Hierarchical chunking: نگه داشتن chunk های کوچک برای retrieval و chunk های بزرگ برای context
  - Document-level chunks: برای اسناد کوتاه

*Embedding Generation:*
- انتخاب مدل embedding: `text-embedding-3-large` از OpenAI، `e5-large-v2`، `bge-m3` (چندزبانه)
- برای ۱۰۰,۰۰۰ سند، batch processing و caching الزامی است
- باید embedding model را بر اساس زبان، domain، و budget انتخاب کنید

*Metadata Enrichment:*
- هر chunk را با metadata غنی کنید: تاریخ ایجاد، نویسنده، دپارتمان، نوع سند، محرمانگی
- این metadata برای filtering در retrieval استفاده می‌شود

**مرحله ۲: Vector Database**

گزینه‌ها و trade-off ها:
- **Pinecone**: Managed service، آسان اما گران. برای شروع سریع مناسب
- **Weaviate**: Open-source، hybrid search (vector + BM25) داخلی، خوب برای production
- **Qdrant**: بسیار سریع، support خوب برای payload filtering، Rust-based
- **pgvector**: اگر PostgreSQL دارید، افزودن vector search بدون infrastructure جدید

برای ۱۰۰,۰۰۰ سند، HNSW (Hierarchical Navigable Small World) ایندکس مناسب است — تقریبی اما سریع و قابل تنظیم از نظر recall/speed tradeoff.

**مرحله ۳: Retrieval Strategy**

*Hybrid Search (بهترین رویکرد):*
Dense retrieval (vector similarity) را با Sparse retrieval (BM25/keyword) ترکیب کنید. Dense خوب است برای semantic similarity، BM25 خوب است برای exact keyword match. Reciprocal Rank Fusion (RRF) برای ترکیب نتایج:

```
RRF(d) = Σ 1/(k + rank(d))
```

*Query Enhancement:*
- **HyDE (Hypothetical Document Embeddings)**: از LLM می‌خواهیم یک سند فرضی بنویسد که پاسخ سوال را دارد، سپس embedding آن را برای retrieval استفاده می‌کنیم
- **Query Expansion**: تولید query های مشابه برای افزایش recall
- **Multi-query retrieval**: تولید چند variation از سوال و merge نتایج

**مرحله ۴: Re-ranking**

پس از retrieval، top-K chunk (مثلاً ۲۰) را با یک Cross-Encoder re-rank می‌کنیم. Cross-Encoder برخلاف Bi-Encoder، query و document را با هم می‌بیند و score دقیق‌تری می‌دهد. مدل‌هایی مثل `cross-encoder/ms-marco-MiniLM-L-6-v2` برای این کار خوب هستند. سپس top-5 را به LLM می‌فرستیم.

**مرحله ۵: Context Assembly و Generation**

- **Context Window Management**: اگر context بزرگ است، از چه ترتیبی برای chunk ها استفاده کنیم؟ "Lost in the Middle" نشان می‌دهد LLM ها اطلاعات وسط context را نادیده می‌گیرند. پس مهم‌ترین chunk ها باید اول و آخر باشند.
- **Citation و Attribution**: هر پاسخ باید مشخص کند از کدام سند آمده — این برای enterprise use case ضروری است
- **Faithfulness Check**: بررسی اینکه آیا پاسخ از context حمایت می‌شود یا hallucination است

**Evaluation Framework:**

برای ارزیابی کیفیت RAG از RAGAS (RAG Assessment) استفاده کنید:
- **Faithfulness**: آیا پاسخ با context سازگار است؟
- **Answer Relevancy**: آیا پاسخ به سوال مرتبط است؟
- **Context Precision**: آیا chunk های retrieve شده مرتبط هستند؟
- **Context Recall**: آیا اطلاعات لازم را retrieve کرده‌ایم؟

**اشتباهات رایج کاندیداها:**
- نادیده گرفتن chunking strategy — این تاثیر بیشتری از embedding model دارد.
- استفاده از فقط vector search بدون keyword search.
- عدم ارزیابی سیستم با benchmark مناسب قبل از production.
- نادیده گرفتن latency — re-ranking گران است و باید caching داشته باشید.

**سوالات follow-up احتمالی:**
- "چطور با اسناد محرمانه کنار می‌آیید؟ آیا user permissions را در retrieval اعمال می‌کنید؟"
- "Fine-tuning LLM در مقابل RAG — چه زمانی کدام را انتخاب می‌کنید؟"
- "چطور سیستم RAG را در production مانیتور می‌کنید؟"

---
### سوال ۷: LLM Fine-tuning — چه زمانی و چگونه
**دسته:** LLM / Fine-tuning
**سطح:** Senior
**چرا مهمه:** Fine-tuning یک ابزار قدرتمند اما گران و پیچیده است. دانستن اینکه چه زمانی واقعاً لازم است یک مهارت کلیدی AI Engineer است.
**زمان پاسخ:** ۲۰ تا ۳۰ دقیقه

**سوال:** چه زمانی fine-tuning یک LLM را به جای prompt engineering یا RAG انتخاب می‌کنید؟ اگر تصمیم به fine-tuning گرفتید، چه رویکردهایی وجود دارد و trade-off های آن‌ها چیست؟

**پاسخ کامل:**

Fine-tuning نه اولین راه‌حل، بلکه آخرین راه‌حل باید باشد. ابتدا باید سایر رویکردها را امتحان کنید.

**هرم تصمیم‌گیری (از ساده به پیچیده):**

۱. **Prompt Engineering**: آیا با یک system prompt خوب می‌توان به نتیجه رسید؟
۲. **Few-Shot Prompting**: آیا چند مثال در prompt کافی است؟
۳. **RAG**: آیا مشکل اصلی دسترسی به دانش خارجی است؟
۴. **Fine-tuning**: اگر هیچ‌کدام کافی نبود

**چه زمانی Fine-tuning واقعاً لازم است:**

- **Tone و Style خاص**: می‌خواهید مدل دقیقاً مثل برند شما صحبت کند و این با prompting کافی نیست
- **فرمت خروجی خاص و پیچیده**: مثلاً تولید همیشگی یک JSON schema پیچیده
- **Latency بهبود یابد**: با fine-tuning می‌توان از مدل کوچک‌تر استفاده کرد که هزینه و latency را کاهش می‌دهد
- **دانش تخصصی domain**: وقتی مدل باید اصطلاحات و مفاهیم یک حوزه بسیار تخصصی را عمیقاً بداند
- **Behavior alignment**: تغییر نحوه رفتار مدل در موقعیت‌های خاص

**تکنیک‌های Fine-tuning:**

*Full Fine-tuning:*
تمام parameters مدل آپدیت می‌شوند. نیاز به GPU memory بسیار زیاد دارد (برای یک مدل ۷B پارامتر، حداقل ۱۴۰GB در fp32 یا ۷۰GB در fp16). برای اکثر use case ها ضروری نیست.

*LoRA (Low-Rank Adaptation):*
به جای آپدیت تمام weight ها، دو ماتریس کم‌رتبه A (d×r) و B (r×d) اضافه می‌کنیم که r بسیار کوچک‌تر از d است:

```
W' = W + BA
```

با r=8 یا r=16، می‌توان تعداد parameters قابل آموزش را تا ۱۰۰۰۰ برابر کاهش داد. **QLoRA** ترکیب LoRA با Quantization (4-bit) است که fine-tuning یک مدل ۶۵B را روی یک GPU 48GB ممکن می‌کند.

*Hyperparameters کلیدی در LoRA:*
- **rank (r)**: رتبه ماتریس‌های پایین رتبه — بزرگ‌تر یعنی capacity بیشتر اما computationally گران‌تر. r=8 یا r=16 شروع خوبی است.
- **alpha**: scaling factor — معمولاً 2×r
- **target_modules**: کدام layers آداپت شوند — معمولاً attention layers (q_proj, v_proj)
- **dropout**: برای regularization

*RLHF و DPO:*
برای alignment از انسان‌ها، **RLHF (Reinforcement Learning from Human Feedback)** استاندارد بود اما پیچیده است. **DPO (Direct Preference Optimization)** جایگزین ساده‌تری است که با داده‌های (chosen, rejected) مستقیماً مدل را align می‌کند بدون نیاز به reward model جداگانه.

**داده Fine-tuning:**

کیفیت داده از کمیت آن مهم‌تر است. برای Supervised Fine-tuning (SFT):
- ۱۰۰۰-۱۰,۰۰۰ مثال با کیفیت بالا معمولاً کافی است
- فرمت chat (instruction/response) باید با مدل base سازگار باشد
- **Catastrophic Forgetting**: با fine-tuning ممکن است مدل capability های قبلی را از دست بدهد. راه‌حل: اضافه کردن چند درصد داده general-purpose به dataset fine-tuning

**Evaluation:**

- Perplexity روی validation set
- Task-specific metrics (مثل ROUGE برای summarization، Exact Match برای QA)
- Human evaluation برای کیفیت کلی
- Safety evaluation — مطمئن شوید fine-tuning alignment را نشکسته

**Cost Estimation:**

برای یک مدل ۷B با LoRA:
- یک A100 80GB GPU: ~$3-4 در ساعت روی cloud
- ۱۰۰۰ مثال + ۳ epoch: شاید ۲-۳ ساعت
- بودجه کل: ~$10-15 برای fine-tuning اولیه

**اشتباهات رایج کاندیداها:**
- رفتن مستقیم به fine-tuning بدون امتحان کردن prompting و RAG.
- Fine-tuning روی داده کم و کیفیت پایین — garbage in, garbage out.
- فراموش کردن evaluation — بدون مقایسه با baseline، نمی‌دانید fine-tuning واقعاً کمک کرده.
- نادیده گرفتن catastrophic forgetting.

**سوالات follow-up احتمالی:**
- "چطور مطمئن می‌شوید fine-tuned model در production drift نمی‌کند؟"
- "اگر داده labeled کافی ندارید، چطور از self-instruct یا synthetic data generation استفاده می‌کنید؟"
- "Difference between instruction tuning و domain-specific fine-tuning؟"

---
### سوال ۸: LLM Serving و Inference Optimization
**دسته:** MLOps / Inference Infrastructure
**سطح:** Staff
**چرا مهمه:** هزینه inference در production می‌تواند از هزینه training بیشتر شود. بهینه‌سازی inference مستقیماً بر profitability تاثیر دارد.
**زمان پاسخ:** ۲۵ تا ۳۵ دقیقه

**سوال:** شما باید یک LLM با ۷۰ میلیارد پارامتر را در production deploy کنید که باید ۱۰۰۰ request در ثانیه را با latency زیر ۵۰۰ms پاسخ دهد. چه تکنیک‌هایی برای inference optimization استفاده می‌کنید؟

**پاسخ کامل:**

این یک چالش واقعی است که production AI teams با آن روبرو هستند. ابتدا باید درک کنیم چرا LLM inference سخت است.

**چرا LLM Inference گران است:**

یک مدل ۷۰B در fp16 نیاز به ۱۴۰GB GPU memory دارد — حداقل ۲ تا A100 80GB. Generation توکن به توکن است (auto-regressive) و inherently sequential. برای هر توکن جدید، باید تمام key-value cache قبلی خوانده شود (memory bandwidth bottleneck).

**تکنیک ۱: Quantization**

کاهش precision محاسبات برای کاهش memory و افزایش throughput:

- **FP16/BF16**: استاندارد — ۵۰% کاهش نسبت به FP32 بدون افت قابل توجه کیفیت
- **INT8**: کاهش ۷۵% memory. با LLM.int8() از bitsandbytes می‌توان outlier features را در FP16 نگه داشت
- **INT4 (GPTQ, AWQ)**: کاهش ۸۷.۵%. GPTQ وزن‌ها را post-training quantize می‌کند با error compensation. AWQ بر اساس activation magnitude، وزن‌های مهم را محافظت می‌کند
- **FP8**: NVIDIA H100 پشتیبانی native دارد — بهترین tradeoff بین speed و quality

برای یک مدل ۷۰B با AWQ به INT4، فقط به یک A100 80GB نیاز دارید به جای دو تا.

**تکنیک ۲: KV Cache Optimization**

در attention mechanism، برای هر توکن قبلی باید Key و Value را محاسبه کنیم. KV Cache این‌ها را ذخیره می‌کند اما memory گران‌قیمت GPU را مصرف می‌کند.

- **PagedAttention (vLLM)**: بزرگ‌ترین نوآوری در LLM serving. KV Cache را مثل memory paging در OS مدیریت می‌کند. به جای allocation حافظه continuous برای هر request، از page-based memory management استفاده می‌کند. این **GPU memory fragmentation** را از ~۶۰% به ~۴% کاهش می‌دهد و throughput را ۲۴× بهبود می‌دهد.
- **Sliding Window Attention**: برای sequence های طولانی، فقط KV cache پنجره‌ای از توکن‌های اخیر را نگه می‌دارد (Mistral)

**تکنیک ۳: Continuous Batching**

در سیستم‌های سنتی، batch پس از اتمام همه request ها پردازش می‌شود. Continuous batching (یا in-flight batching) اجازه می‌دهد request های جدید در وسط پردازش batch اضافه شوند. این GPU utilization را از ~۳۰% به ~۸۰%+ می‌رساند.

**تکنیک ۴: Speculative Decoding**

برای کاهش latency:
- یک مدل کوچک (draft model) چند توکن پیشنهاد می‌دهد
- مدل اصلی (target model) این پیشنهادات را verify می‌کند
- اگر پیشنهادات درست باشند، چند توکن به جای یک توکن در یک pass تولید می‌شوند
- می‌تواند latency را ۲-۳ برابر کاهش دهد بدون از دست دادن کیفیت

**تکنیک ۵: Tensor Parallelism و Pipeline Parallelism**

برای توزیع مدل روی چند GPU:
- **Tensor Parallelism**: هر layer روی چند GPU به صورت موازی محاسبه می‌شود. مناسب برای کاهش latency اما نیاز به communication bandwidth بالا دارد (NVLink).
- **Pipeline Parallelism**: layers مختلف روی GPU های مختلف. مناسب برای throughput اما bubble overhead دارد.
- **Sequence Parallelism**: sequence را روی GPU ها تقسیم می‌کنیم.

**معماری پیشنهادی برای ۱۰۰۰ RPS:**

```
Load Balancer (L7)
    ↓
[Router + Request Queue]
    ↓
vLLM Serving Cluster
- 4x Node (هر node: 8x H100 80GB)
- AWQ INT4 quantization
- Continuous batching
- Speculative decoding با draft model 7B
    ↓
[Response Cache (Redis)] ← [Semantic Cache]
```

- **Semantic Cache**: embedding برای هر query، اگر سوال مشابه قبلاً پرسیده شده cache return می‌کند — می‌تواند ۳۰-۵۰% از requests را بدون GPU پاسخ دهد
- Horizontal scaling آسان با stateless serving nodes

**ابزارهای رایج:**
- **vLLM**: بهترین throughput، PagedAttention، Continuous batching
- **TensorRT-LLM**: از NVIDIA، بیشترین optimization برای H100
- **TGI (Text Generation Inference)**: از HuggingFace، آسان برای deploy
- **Triton Inference Server**: برای multi-model serving

**اشتباهات رایج کاندیداها:**
- نادیده گرفتن memory bandwidth bottleneck — اکثر LLM serving memory-bound است نه compute-bound.
- عدم استفاده از Continuous Batching — بزرگترین single improvement معمولاً همین است.
- فراموش کردن monitoring: latency percentiles (P50, P99)، token throughput، GPU utilization، cache hit rate.

**سوالات follow-up احتمالی:**
- "چطور cost per token را برای تصمیمات business محاسبه می‌کنید؟"
- "Prefill vs Decode phase چه تفاوتی در resource utilization دارند و چطور برای هر کدام بهینه می‌کنید؟"
- "Disaggregated Prefill-Decode چیست و چه مزایایی دارد؟"

---
### سوال ۹: Prompt Engineering پیشرفته و Chain-of-Thought
**دسته:** Prompt Engineering / LLM Behavior
**سطح:** Mid/Senior
**چرا مهمه:** Prompt engineering یک مهارت اساسی برای هر AI Engineer است. تفاوت بین یک prompt خوب و بد می‌تواند accuracy را ۳۰-۴۰% تغییر دهد.
**زمان پاسخ:** ۱۵ تا ۲۵ دقیقه

**سوال:** تکنیک‌های پیشرفته prompt engineering را توضیح دهید. چطور یک LLM را برای حل مسائل پیچیده استدلالی آموزش می‌دهید؟ چه زمانی Chain-of-Thought واقعاً کمک می‌کند؟

**پاسخ کامل:**

Prompt Engineering در سال‌های اخیر از یک هنر غیررسمی به یک discipline با تکنیک‌های اثبات‌شده تبدیل شده است.

**تکنیک ۱: Zero-Shot vs Few-Shot Prompting**

*Zero-Shot*: فقط دستورالعمل بدون مثال. مناسب برای وظایف ساده که مدل به خوبی روی آن‌ها fine-tune شده.

*Few-Shot*: چند مثال (معمولاً ۳-۵) از input/output مطلوب. برای وظایفی که مدل باید format یا style خاصی را یاد بگیرد. **کیفیت مثال‌ها از تعداد آن‌ها مهم‌تر است.** نظم قرار دادن مثال‌ها نیز اهمیت دارد — آخرین مثال بیشترین تاثیر را دارد.

**تکنیک ۲: Chain-of-Thought (CoT)**

CoT مدل را مجبور می‌کند قبل از رسیدن به پاسخ، مراحل استدلال را صریحاً بنویسد. این به ویژه برای:
- مسائل ریاضی و منطقی
- سوالات چند مرحله‌ای
- وظایفی که نیاز به integration اطلاعات از چند منبع دارند

کارا است. تحقیقات نشان می‌دهد برای مسائل ریاضی روی مدل‌های بزرگ، CoT دقت را از ~۱۷٪ به ~۶۹٪ می‌رساند.

```
# بدون CoT
"چند شترمرغ در ۳ گله با ۳۴، ۲۸ و ۴۵ شترمرغ وجود دارد؟"
→ ممکن است اشتباه جواب دهد

# با CoT  
"بیایید قدم به قدم حل کنیم:
گله اول: ۳۴
گله دوم: ۲۸
گله سوم: ۴۵
مجموع: ۳۴ + ۲۸ + ۴۵ = ۱۰۷"
→ دقت بالاتر
```

CoT برای مدل‌های کوچک (زیر ۱۰۰B پارامتر) کمتر موثر است — به نظر می‌رسد یک توانایی emergent در مدل‌های بزرگ است.

**تکنیک ۳: Tree-of-Thoughts (ToT)**

به جای یک زنجیره خطی استدلال، چندین مسیر را به صورت موازی بررسی می‌کنیم. مدل "افکار" مختلف تولید می‌کند، هر کدام را ارزیابی می‌کند (با خود مدل یا یک evaluator جداگانه)، و بهترین مسیر را دنبال می‌کند. برای مسائل برنامه‌ریزی و jigsaw puzzle ها بسیار موثر است.

**تکنیک ۴: ReAct (Reasoning + Acting)**

ترکیب استدلال با tool use:
```
Thought: باید قیمت طلا را بررسی کنم
Action: search("قیمت طلا امروز")
Observation: [نتیجه search]
Thought: حالا می‌توانم پاسخ بدهم
Answer: ...
```

این معماری پایه برای اکثر AI Agents است و به مدل اجازه می‌دهد با محیط تعامل کند.

**تکنیک ۵: Self-Consistency**

به جای یک بار generation، چند بار (مثلاً ۱۰ بار) با temperature > 0 generate می‌کنیم و majority vote می‌گیریم. این accuracy را به قیمت compute بیشتر بهبود می‌دهد. برای مسائل با پاسخ قطعی مثل ریاضیات بهترین نتیجه را دارد.

**تکنیک ۶: Meta-Prompting**

از LLM می‌خواهیم prompt بهتری برای همان مسئله بسازد. یا از LLM می‌خواهیم پاسخ خودش را critique و بهبود دهد (Constitutional AI، Self-Refine).

**Prompt Structure بهینه:**

```
[Role/Persona]: "شما یک متخصص حقوقی هستید..."
[Context]: اطلاعات مرتبط با سوال
[Task]: وظیفه دقیق
[Constraints]: محدودیت‌ها (فارسی بنویس، حداکثر ۲۰۰ کلمه)
[Output Format]: فرمت دقیق خروجی
[Examples] (اختیاری): چند مثال
```

**Prompt Injection Attacks:**

در production باید از prompt injection محافظت کنید — کاربر ممکن است با دستکاری input سعی کند system prompt را دور بزند. راه‌حل‌ها:
- جداسازی واضح user input از system instructions
- Input validation و sanitization
- Sandboxing با output validation

**اشتباهات رایج کاندیداها:**
- گفتن "فقط از CoT استفاده کنید" بدون توجه به اینکه CoT برای وظایف ساده overhead غیرضروری اضافه می‌کند.
- نادیده گرفتن temperature tuning — برای reasoning tasks temperature پایین (0.1-0.3) بهتر است.
- عدم ارزیابی prompt به صورت سیستماتیک با benchmark.

**سوالات follow-up احتمالی:**
- "چطور prompts خود را ارزیابی می‌کنید؟ چه evaluation framework ای استفاده می‌کنید؟"
- "Prompt caching چیست و چطور cost را کاهش می‌دهد؟"
- "چطور از prompt injection جلوگیری می‌کنید وقتی user input در system prompt قرار می‌گیرد؟"

---
### سوال ۱۰: ارزیابی LLM — Evaluation Framework های Production
**دسته:** LLM Evaluation / Quality Assurance
**سطح:** Senior/Staff
**چرا مهمه:** "You can't improve what you can't measure." بدون evaluation دقیق، نمی‌دانید آیا تغییرات prompt، fine-tuning، یا RAG واقعاً بهتر شده‌اند.
**زمان پاسخ:** ۲۰ تا ۳۰ دقیقه

**سوال:** چطور یک framework جامع برای ارزیابی یک LLM-based chatbot که سوالات فنی کاربران را پاسخ می‌دهد طراحی می‌کنید؟ از چه ترکیبی از automated و human evaluation استفاده می‌کنید؟

**پاسخ کامل:**

ارزیابی LLM یکی از دشوارترین چالش‌های AI Engineering است. برخلاف classification که F1-Score کافی است، ارزیابی generation نیازمند چندین لایه از متریک است.

**ابعاد ارزیابی:**

هر LLM-based system باید از چند بُعد ارزیابی شود:

۱. **Correctness**: آیا پاسخ از نظر فنی درست است؟
۲. **Faithfulness**: آیا ادعاها با منابع (در RAG) سازگار هستند؟
۳. **Relevance**: آیا پاسخ به سوال مرتبط است؟
۴. **Completeness**: آیا همه جنبه‌های سوال پوشش داده شده؟
۵. **Clarity**: آیا پاسخ قابل فهم است؟
۶. **Safety**: آیا محتوای مضر یا inappropriate وجود دارد؟

**لایه ۱: Unit Tests و Regression Tests**

مشابه software testing، یک test suite از سوالات با پاسخ مرجع بسازید:

```python
test_cases = [
    {
        "question": "Python's GIL چیست؟",
        "expected_keywords": ["Global Interpreter Lock", "threading", "multiprocessing"],
        "must_not_contain": ["JavaScript", "Java"],
        "category": "python_fundamentals"
    }
]
```

این test ها باید با هر تغییر در مدل، prompt، یا RAG pipeline اجرا شوند. Regression testing مطمئن می‌شود بهبود در یک حوزه باعث خرابی در حوزه دیگر نشده.

**لایه ۲: LLM-as-Judge (Automated Evaluation)**

از یک LLM قوی‌تر (مثل GPT-4 یا Claude Opus) به عنوان judge استفاده می‌کنیم. این رویکرد scalable است اما باید با human evaluation validate شود.

```python
judge_prompt = """
شما یک ارزیاب تخصصی هستید. سوال، پاسخ مرجع، و پاسخ مدل را بررسی کنید.

سوال: {question}
پاسخ مرجع: {reference}
پاسخ مدل: {model_answer}

لطفاً پاسخ را از ۱ تا ۵ برای هر معیار ارزیابی کنید:
- Correctness: {score}
- Completeness: {score}  
- Clarity: {score}
دلیل: {explanation}
"""
```

معیارهای کلیدی برای LLM-as-Judge:
- **Pairwise comparison**: به judge دو پاسخ نشان می‌دهیم و می‌پرسیم کدام بهتر است — این reliable‌تر از absolute scoring است
- **G-eval**: استفاده از CoT در judge برای دقت بیشتر
- **Calibration**: مقایسه judge scores با human labels برای اطمینان از alignment

**لایه ۳: متریک‌های Reference-based**

وقتی پاسخ مرجع داریم:
- **ROUGE**: برای summarization — recall مبتنی بر overlap n-gram
- **BLEU**: برای translation — precision مبتنی بر n-gram
- **BERTScore**: cosine similarity بین embeddings — بهتر از ROUGE برای capture کردن semantic similarity
- **Exact Match (EM)**: برای سوالات با پاسخ کوتاه

**لایه ۴: Behavioral Testing (Black-box Probing)**

- **Adversarial Examples**: ایجاد سوالات دشوار طراحی‌شده برای فریب مدل
- **Edge Cases**: سوالات خارج از distribution عادی
- **Consistency Testing**: پرسیدن سوال یکسان به شکل‌های مختلف — آیا پاسخ consistent است؟
- **Counterfactual Testing**: تغییر جزئی سوال — آیا پاسخ به درستی تغییر می‌کند؟

**لایه ۵: Human Evaluation**

با اینکه گران است، human evaluation ground truth نهایی است:
- **A/B Testing**: ارزیابان بدون دانستن کدام مدل، دو پاسخ را مقایسه می‌کنند
- **Absolute Rating**: هر پاسخ جداگانه از ۱ تا ۵ rate می‌شود
- **Annotation Guidelines**: دستورالعمل دقیق برای ارزیابان — inter-annotator agreement (Cohen's Kappa) باید > 0.7 باشد
- **Representative Sampling**: نمونه باید تمام categories سوالات را پوشش دهد

**Online Evaluation (Production Monitoring):**

- **Implicit Feedback**: آیا کاربر پس از دریافت پاسخ سوال follow-up می‌پرسد؟ آیا thumbs up/down می‌زند؟
- **Session Success**: آیا کاربر به نتیجه مطلوب رسید؟
- **Latency و Error Rate**: متریک‌های infrastructure

**ابزارهای evaluation:**
- **LangSmith** (LangChain): ردیابی و ارزیابی pipeline
- **Weights & Biases Evaluation**: integration با W&B
- **RAGAS**: برای RAG-specific evaluation
- **DeepEval**: framework open-source برای LLM evaluation

**اشتباهات رایج کاندیداها:**
- اعتماد به تنها یک متریک — ترکیب چند متریک لازم است.
- نادیده گرفتن online evaluation — production feedback مهم‌ترین signal است.
- عدم ایجاد golden dataset برای regression testing.
- فراموش کردن evaluation زمانی که تیم تحت فشار delivery است.

**سوالات follow-up احتمالی:**
- "چطور با پدیده 'verbosity bias' در LLM-as-judge کنار می‌آیید (تمایل judge به پاسخ‌های طولانی‌تر)؟"
- "چطور evaluation pipeline خود را برای multi-turn conversations adapt می‌کنید؟"
- "Red teaming چیست و چطور در evaluation framework جای می‌گیرد؟"

---

## بخش سوم: DL/ML System Design

---
### سوال ۱۱: طراحی سیستم توصیه‌گر Real-time
**دسته:** Recommendation System / System Design
**سطح:** Staff/Principal
**چرا مهمه:** Recommendation System ها در Netflix، YouTube، Amazon و اکثر پلتفرم‌های بزرگ بیشترین revenue را تولید می‌کنند. طراحی آن‌ها در مقیاس بزرگ یک مهارت کلیدی Staff Engineer است.
**زمان پاسخ:** ۴۰ تا ۶۰ دقیقه

**سوال:** برای یک پلتفرم streaming (مثل Netflix) با ۲۰۰ میلیون کاربر و ۱۵,۰۰۰ عنوان محتوا، یک سیستم توصیه‌گر real-time طراحی کنید. باید در کمتر از ۱۰۰ms پاسخ دهد.

**پاسخ کامل:**

این یک سوال classic System Design ML است. پاسخ باید هم معماری کلی سیستم را پوشش دهد هم جزئیات الگوریتمی را.

**تعریف مسئله و Clarification:**

ابتدا باید سوال‌های clarification بپرسیم:
- آیا personalized recommendation می‌خواهیم یا general trending؟
- چه سیگنال‌هایی داریم؟ (explicit ratings، implicit feedback مثل watch time)
- Cold start problem چطور مدیریت می‌شود (کاربران جدید، محتوای جدید)؟
- آیا real-time feature update نیاز داریم یا batch کافی است؟

**معماری کلی — Two-Stage Recommendation:**

اکثر سیستم‌های بزرگ از یک معماری دو مرحله‌ای استفاده می‌کنند:

```
[۲۰۰M کاربر × ۱۵,۰۰۰ محتوا] → [Candidate Generation] → [~۱۰۰۰ کاندیدا] → [Ranking] → [Top-K Results]
```

**مرحله ۱: Candidate Generation (Retrieval)**

هدف: از ۱۵,۰۰۰ محتوا، ۱۰۰۰-۲۰۰۰ کاندیدا را سریع فیلتر کنیم.

*Collaborative Filtering — Matrix Factorization:*
User-Item interaction matrix را به دو ماتریس کم‌رتبه تجزیه می‌کنیم:
```
R ≈ U × Iᵀ
```
که U ماتریس user embeddings و I ماتریس item embeddings است. با ALS (Alternating Least Squares) یا SGD آموزش می‌بینند. پس از آموزش، برای هر کاربر، نزدیک‌ترین item ها را با Approximate Nearest Neighbor (FAISS, ScaNN) پیدا می‌کنیم.

*Two-Tower Network:*
یک شبکه دو برج که user tower و item tower به صورت جداگانه embedding تولید می‌کنند و در پایان با dot product مشابهت محاسبه می‌شود. مزیت: item embeddings را می‌توان pre-compute کرد و در ANN index ذخیره کرد.

*Content-Based Filtering:*
بر اساس ویژگی‌های محتوا (genre، cast، director، tags) و ویژگی‌های کاربر (تاریخچه بازدید). مفید برای cold start.

*Real-time Signals برای Retrieval:*
سشن جاری کاربر (آخرین چند محتوایی که دیده) برای context-aware retrieval استفاده می‌شود.

**مرحله ۲: Ranking**

از ۱۰۰۰ کاندیدا، ۱۰-۲۰ توصیه نهایی را انتخاب می‌کنیم. این مرحله می‌تواند از feature های غنی‌تر استفاده کند:

*Feature های Ranking:*
- **User features**: demographics، تاریخچه طولانی‌مدت، preferences
- **Item features**: محبوبیت، freshness، genre، مدت زمان
- **Context features**: زمان روز، device، روز هفته
- **User-Item interaction features**: آیا کاربر قبلاً این نوع محتوا دیده؟

*مدل Ranking:*
- **GBDT (Gradient Boosted Decision Trees)**: سریع، interpretable، خوب برای tabular features
- **Deep & Cross Network (DCN)**: ترکیب deep learning با feature crossing صریح
- **Multi-task Learning**: همزمان بهینه‌سازی چند objective (کلیک، watch time، rating)

*Multi-objective Optimization:*
نباید فقط CTR (Click-Through Rate) را بهینه کنیم — این به click-bait منجر می‌شود. باید objective های چندگانه را ترکیب کنیم:
```
Score = w₁ × P(click) + w₂ × P(watch_completion) + w₃ × P(explicit_positive)
```

**مرحله ۳: Re-ranking و Business Rules**

- **Diversity**: اطمینان از اینکه توصیه‌ها از genre های مختلف هستند
- **Freshness**: دادن boost به محتوای جدید
- **Context-awareness**: توصیه‌های متفاوت برای زمان شب (آرام) در مقابل صبح (کوتاه)
- **Business constraints**: محتوای sponsored، محتوای regional

**Architecture در Production:**

```
User Request (100ms budget)
    ↓ [10ms] Feature Fetching (Redis/Feature Store)
    ↓ [30ms] Candidate Generation (FAISS ANN + pre-computed)
    ↓ [40ms] Ranking Model (GPU/TensorRT)  
    ↓ [10ms] Re-ranking + Business Rules
    ↓ [10ms] Response Assembly
```

*Feature Store:*
- **Online Store** (Redis): user features آپدیت شده در real-time
- **Offline Store** (S3/BigQuery): features batch-computed

*Pre-computation:*
Item embeddings را روزانه pre-compute می‌کنیم و در FAISS index نگه می‌داریم — این retrieval را به ۱۰ms می‌رساند.

**Training Pipeline:**

- **Batch Training**: مدل ranking هر روز با داده ۳۰ روز گذشته آموزش می‌بیند
- **Online Learning**: برای real-time adaptation، lightweight model با incremental updates
- **Exploration vs Exploitation**: Epsilon-greedy یا UCB برای نشان دادن محتوای جدید (exploration)

**Evaluation:**

- **Offline**: nDCG (Normalized Discounted Cumulative Gain)، Precision@K، Recall@K
- **Online**: A/B Test روی watch time، retention، engagement

**Cold Start Problem:**

- **کاربر جدید**: توصیه‌های محبوب در سطح کشور + onboarding سریع (چند سوال اولیه)
- **محتوای جدید**: تزریق محتوا در exploration traffic، استفاده از content-based features

**اشتباهات رایج کاندیداها:**
- فراموش کردن two-stage architecture — سعی کردن با یک مدل ۲۰۰M × ۱۵,۰۰۰ را به صورت مستقیم rank کنیم.
- نادیده گرفتن cold start.
- بهینه کردن فقط برای CTR (click-bait problem).
- فراموش کردن latency budget — هر مرحله باید سهم خود را از ۱۰۰ms بداند.

**سوالات follow-up احتمالی:**
- "چطور با feedback loop (Filter Bubble) کنار می‌آیید — وقتی مدل فقط محتوایی را توصیه می‌کند که مشابه قبلی است؟"
- "چطور fairness را در recommendation system تضمین می‌کنید؟"
- "اگر real-time personalization نداشتید، چطور با user context خوش‌بینانه عمل می‌کردید؟"

---
### سوال ۱۲: Feature Store — طراحی و معماری
**دسته:** MLOps / Data Infrastructure
**سطح:** Senior/Staff
**چرا مهمه:** Feature Store یکی از حیاتی‌ترین infrastructure های ML در production است. بدون آن، هر تیم feature های خود را جداگانه می‌سازد و training-serving skew اجتناب‌ناپذیر است.
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** شرکت شما ۵۰ Data Scientist دارد که روی ۲۰ مدل مختلف کار می‌کنند. اغلب feature های مشابه را جداگانه می‌سازند و training/serving skew وجود دارد. یک Feature Store طراحی کنید.

**پاسخ کامل:**

Feature Store یک platform مرکزی برای ذخیره، مدیریت، و سرویس‌دهی feature های ML است. مشکلاتی که حل می‌کند:

- **Feature Reuse**: هر Data Scientist feature های آماده را reuse می‌کند نه از صفر می‌سازد
- **Training-Serving Skew**: feature های دقیقاً یکسانی در training و serving استفاده می‌شوند
- **Point-in-Time Correctness**: برای training، feature های لحظه پیش‌بینی استفاده می‌شود نه لحظه build
- **Feature Discoverability**: Data Scientist ها می‌توانند feature های موجود را بررسی کنند

**معماری Feature Store:**

```
[Data Sources] → [Feature Engineering] → [Feature Store] → [Training] / [Serving]
                                              ↑                ↓
                                         [Registry]      [Online Store + Offline Store]
```

**لایه ۱: Feature Registry (Metadata Store)**

مرکز control. هر feature با metadata کامل ثبت می‌شود:

```python
from feast import Feature, FeatureView, Entity

# تعریف Entity
user = Entity(name="user_id", value_type=ValueType.INT64)

# تعریف Feature View
user_features = FeatureView(
    name="user_activity_features",
    entities=["user_id"],
    ttl=Duration(seconds=86400),  # 24 ساعت
    features=[
        Feature(name="total_purchases_7d", dtype=ValueType.INT64),
        Feature(name="avg_session_duration_30d", dtype=ValueType.FLOAT),
        Feature(name="last_login_days_ago", dtype=ValueType.INT64),
    ],
    online=True,  # در online store هم ذخیره شود
    batch_source=BigQuerySource(table_ref="project.dataset.user_features"),
)
```

Feature Registry شامل:
- نام و schema هر feature
- نحوه محاسبه (transformation logic)
- منبع داده
- owner و تاریخ ایجاد
- SLA های freshness
- versioning و lineage

**لایه ۲: Offline Store**

برای training و batch processing:
- **ابزار**: Hive، BigQuery، Snowflake، یا Delta Lake
- **Point-in-Time Joins**: مهم‌ترین feature در offline store. وقتی training data را آماده می‌کنید، باید feature های لحظه event (نه لحظه حال) را برگردانید.

```sql
-- Point-in-Time Join
SELECT events.user_id, events.timestamp, events.label,
       features.total_purchases_7d,  -- این مقدار را از لحظه events.timestamp می‌گیریم
FROM events
JOIN features
ON events.user_id = features.user_id
AND features.feature_timestamp <= events.timestamp
```

اگر این join درست انجام نشود، **future leakage** رخ می‌دهد — مدل از feature های آینده استفاده می‌کند که در serving در دسترس نیستند.

**لایه ۳: Online Store**

برای real-time serving با latency بسیار پایین (< 5ms):
- **ابزار**: Redis، DynamoDB، Cassandra، Bigtable
- Feature های آپدیت‌شده به صورت real-time (push) یا scheduled (pull) به online store می‌رسند
- Lookup با key = entity_id (مثلاً user_id)

**لایه ۴: Feature Computation Pipeline**

Feature ها چطور محاسبه می‌شوند:

*Batch Features* — با Spark یا dbt روزانه یا ساعتی update می‌شوند:
```python
# مثال: total purchases در ۷ روز گذشته
df_features = events.filter(
    col("timestamp") >= current_timestamp - interval("7 days")
).groupBy("user_id").agg(count("purchase_id").alias("total_purchases_7d"))
```

*Streaming Features* — با Kafka + Flink یا Spark Streaming به صورت real-time:
- مثال: تعداد کلیک در ۱۰ دقیقه گذشته
- برای on-demand fraud detection یا real-time recommendation

*On-demand Features* — feature هایی که در serving time محاسبه می‌شوند چون به داده‌های request وابسته‌اند:
- مثال: اختلاف قیمت محصول مشاهده‌شده با متوسط قیمت این category برای این کاربر

**Feature Freshness و SLA:**

هر feature یک freshness requirement دارد:
- `last_login_timestamp`: باید real-time باشد (streaming)
- `total_purchases_30d`: می‌تواند روزانه update شود (batch)
- `user_country`: هفتگی کافی است

**مانیتورینگ Feature Store:**

- **Feature Freshness**: آیا feature ها طبق SLA آپدیت می‌شوند؟
- **Training-Serving Skew Detection**: توزیع feature در training vs serving را مقایسه کنید
- **Data Quality**: null rate، outliers، distribution shift

```python
# Training-Serving Skew Detection
training_dist = offline_features["total_purchases_7d"].describe()
serving_dist = online_features_sample["total_purchases_7d"].describe()
skew_metric = PSI(training_dist, serving_dist)  # Population Stability Index
```

**ابزارهای Feature Store:**
- **Feast**: open-source، محبوب، خوب برای شروع
- **Tecton**: managed service، enterprise-grade، streaming support خوب
- **Vertex AI Feature Store**: Google Cloud، tight integration با GCP
- **Databricks Feature Store**: برای تیم‌های Databricks

**اشتباهات رایج کاندیداها:**
- فراموش کردن Point-in-Time Correctness — این رایج‌ترین source of training-serving skew است.
- نادیده گرفتن Feature Freshness SLA.
- عدم طراحی برای feature reuse — نمی‌دانند چطور discoverability را پیاده‌سازی کنند.
- فکر کردن که Feature Store فقط یک database است — بخش اصلی feature computation pipeline است.

**سوالات follow-up احتمالی:**
- "چطور با backfilling feature های جدید برای training داده‌های تاریخی کنار می‌آیید؟"
- "اگر یک feature critical شکست بخورد و null شود، چه fallback strategy ای دارید؟"
- "چطور feature ownership و governance را در یک تیم بزرگ مدیریت می‌کنید؟"

---
### سوال ۱۳: Model Monitoring و Data/Model Drift Detection
**دسته:** MLOps / Production ML
**سطح:** Senior/Staff
**چرا مهمه:** یک مدل که روز اول خوب بود، ممکن است پس از چند هفته در production دیگر خوب نباشد. Monitoring مداوم تفاوت بین ML خوب و ML شکست‌خورده است.
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک مدل credit scoring که ۶ ماه پیش deploy شده را مانیتور می‌کنید. متوجه می‌شوید accuracy مدل در ماه اخیر افت کرده. چه سیستمی برای تشخیص زودهنگام این مشکل طراحی می‌کنید؟

**پاسخ کامل:**

Model Degradation در production یک مشکل ناگزیر است. سوال این نیست که "آیا مدل خراب می‌شود؟" بلکه "چه زمانی و چطور بفهمیم خراب شده؟"

**انواع Drift:**

درک انواع drift برای طراحی monitoring system ضروری است:

**۱. Data Drift (Covariate Shift):**
توزیع feature های ورودی (X) تغییر کرده. مثلاً اگر اقتصاد تغییر کند، توزیع income، debt-to-income ratio تغییر می‌کند.

**۲. Label Drift (Prior Probability Shift):**
توزیع target variable (Y) تغییر کرده. مثلاً نرخ default در اقتصاد نامساعد بالاتر می‌رود.

**۳. Concept Drift:**
رابطه بین X و Y تغییر کرده. یعنی اگر همان feature های قدیمی را داشتید، پاسخ درست متفاوت است. مثلاً یک رفتار مالی که قبلاً indicator خوبی از creditworthiness بود، دیگر نیست.

**۴. Model Quality Degradation:**
نتیجه نهایی — performance مدل واقعاً افت کرده.

**معماری Monitoring System:**

```
[Production Predictions] → [Monitoring Engine] → [Alert System] → [Investigation Dashboard]
        ↓                         ↓
[Ground Truth Labels]     [Drift Detectors]
(با تاخیر)              [Performance Metrics]
```

**لایه ۱: Data Distribution Monitoring**

برای هر feature به صورت روزانه یا ساعتی:

*Statistical Tests برای Drift Detection:*
- **Population Stability Index (PSI)**:
  ```
  PSI = Σ (Actual_% - Expected_%) × ln(Actual_% / Expected_%)
  ```
  PSI < 0.1: بدون drift، PSI 0.1-0.2: متوسط، PSI > 0.2: drift قابل توجه

- **Kolmogorov-Smirnov Test (KS Test)**: برای متغیرهای پیوسته — آیا دو توزیع از یک population هستند؟
- **Chi-Square Test**: برای متغیرهای categorical
- **Jensen-Shannon Divergence**: معیار متقارن از information theory

*برای embedding ها:*
- Cosine similarity بین میانگین embedding فعلی و baseline
- Maximum Mean Discrepancy (MMD)

**لایه ۲: Prediction Monitoring**

حتی بدون label های واقعی می‌توان prediction ها را مانیتور کرد:
- توزیع prediction scores (probability distribution)
- نسبت predictions بالا / پایین یک threshold
- اگر توزیع scores تغییر کند، sign of drift است

**لایه ۳: Business Metric Monitoring**

در credit scoring، حتی بدون دانستن اینکه مدل اشتباه کرده:
- **Approval Rate**: اگر unexpectedly بالا یا پایین رفت، نشانه مشکل است
- **Override Rate**: اگر business analysts بیشتر از حد معمول predictions را override می‌کنند
- **Early Default Rate**: درصد وام‌های تایید شده که زود default کردند

**لایه ۴: Ground Truth Monitoring (با تاخیر)**

برای credit scoring، ground truth (default یا non-default) با ۳-۶ ماه تاخیر مشخص می‌شود. اما می‌توان:
- **Early signals**: تاخیر در پرداخت‌های اول
- **Proxy labels**: تغییر credit score کاربر در ۳۰ روز اول

پس از دریافت ground truth، کامل‌ترین evaluation را داریم:
- AUC-ROC در sliding time window
- Calibration drift (آیا predicted probabilities هنوز calibrate هستند؟)
- Performance در segment های مختلف (geographic، demographic)

**Alerting System:**

```python
class DriftMonitor:
    def check_psi(self, feature_name, baseline, current):
        psi = calculate_psi(baseline, current)
        if psi > 0.25:
            alert(severity="CRITICAL", message=f"Severe drift in {feature_name}: PSI={psi:.3f}")
        elif psi > 0.1:
            alert(severity="WARNING", message=f"Moderate drift in {feature_name}: PSI={psi:.3f}")
    
    def check_performance(self, recent_auc, baseline_auc, threshold=0.03):
        if baseline_auc - recent_auc > threshold:
            alert(severity="CRITICAL", message=f"AUC dropped by {baseline_auc - recent_auc:.3f}")
```

**Canary Deployment برای مدل جدید:**

پس از retrain، مدل جدید را به ۵% ترافیک بفرستید و مقایسه کنید. اگر metrics بهتر بود، rollout کنید.

**Shadow Mode:**

مدل جدید را موازی با مدل قدیم اجرا کنید بدون اینکه تاثیری بر prediction داشته باشد. این به جمع‌آوری real-world performance data قبل از rollout کمک می‌کند.

**Retraining Strategy:**

- **Scheduled Retraining**: مثلاً ماهانه با sliding window از داده‌های جدید
- **Trigger-based Retraining**: وقتی PSI > 0.2 یا performance drop > 3%
- **Online Learning**: برای برخی use case ها، مدل به صورت continuous با داده‌های جدید آپدیت می‌شود (اما نیاز به دقت بیشتری دارد)

**ابزارهای monitoring:**
- **Evidently AI**: open-source، dashboard زیبا برای data/model drift
- **WhyLabs**: cloud-based، ML-specific monitoring
- **Arize AI**: enterprise-grade، real-time monitoring
- **MLflow + Grafana**: self-hosted solution

**اشتباهات رایج کاندیداها:**
- فقط مانیتور کردن technical metrics (latency، error rate) و نه model quality.
- نادیده گرفتن تاخیر در دریافت ground truth.
- عدم بررسی drift در segment های مختلف (یک segment ممکن است دچار drift شود نه کل جمعیت).
- عدم داشتن پروتکل واضح برای "چه زمانی retrain کنیم؟"

**سوالات follow-up احتمالی:**
- "چطور مطمئن می‌شوید که monitoring system خودش به درستی کار می‌کند؟ (monitoring the monitor)"
- "اگر drift در یک segment خاص (مثلاً یک استان) باشد، چطور آن را شناسایی می‌کنید؟"
- "Catastrophic drift در مقابل gradual drift — استراتژی متفاوتی نیاز دارد؟"

---
### سوال ۱۴: Training Pipeline در مقیاس بزرگ — Distributed Training
**دسته:** Distributed Systems / Deep Learning Infrastructure
**سطح:** Staff
**چرا مهمه:** آموزش مدل‌های بزرگ (LLM، Foundation Model) نیازمند درک عمیق distributed training است. این یک مهارت نادر و بسیار ارزشمند است.
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** می‌خواهید یک مدل Language Model با ۱۳۰ میلیارد پارامتر را روی ۱۰۰۰ GPU آموزش دهید. چه معماری distributed training طراحی می‌کنید؟ trade-off های مختلف rویکردها چیست؟

**پاسخ کامل:**

آموزش مدل‌های بزرگ یکی از پیچیده‌ترین مشکلات مهندسی در AI است. یک مدل ۱۳۰B در bf16 نیاز به ۲۶۰GB GPU memory دارد — در یک GPU 80GB جا نمی‌گیرد. باید از parallelism استفاده کنیم.

**چهار نوع Parallelism:**

**۱. Data Parallelism (DP)**

هر GPU یک کپی کامل از مدل دارد اما batch مختلف:
- Forward و backward pass به صورت موازی
- Gradient ها aggregated می‌شوند (All-Reduce)
- **محدودیت**: مدل باید در یک GPU جا بشود
- **مناسب**: وقتی مدل کوچک است و می‌خواهیم throughput را با batch size بزرگ‌تر افزایش دهیم

**ZeRO (Zero Redundancy Optimizer):**
از DeepSpeed، memory را بین GPU های DP تقسیم می‌کند بدون افزایش communication:
- **ZeRO-1**: Optimizer states را partition می‌کند (4x کاهش memory)
- **ZeRO-2**: ZeRO-1 + Gradient partitioning (8x کاهش)
- **ZeRO-3**: ZeRO-2 + Parameter partitioning (theoretically unlimited scale)

**۲. Tensor Parallelism (TP)**

هر layer به صورت موازی روی چند GPU اجرا می‌شود:
- Matrix multiplication را می‌توان به صورت row-wise یا column-wise تقسیم کرد
- نیاز به All-Reduce بین هر layer — communication overhead قابل توجه
- **مناسب**: GPU های با NVLink پرسرعت (درون یک node)
- معمولاً TP-degree ≤ 8 (درون یک node با 8 GPU)

```
# Linear Layer: Y = XW + b
# W را به W1 و W2 تقسیم می‌کنیم روی GPU های مختلف
GPU_1: Y_partial_1 = X @ W1
GPU_2: Y_partial_2 = X @ W2
All-Reduce: Y = concatenate(Y_partial_1, Y_partial_2)
```

**۳. Pipeline Parallelism (PP)**

layer های مختلف روی GPU های مختلف:
- GPU_1: Layer 1-10, GPU_2: Layer 11-20, ...
- **Bubble Problem**: در naive pipeline، GPU_1 باید صبر کند تا GPU_N forward pass را تمام کند
- **Micro-batching**: batch را به micro-batch های کوچک تقسیم کنیم تا GPU های بعدی زودتر شروع کنند
- GPipe و PipeDream روش‌های متفاوت برای کاهش bubble overhead دارند

**۴. Sequence Parallelism (SP)**

برای attention با sequence طولانی، sequence را بین GPU ها تقسیم کنیم:
- Memory complexity attention: O(N²) که N طول sequence است
- با SP، هر GPU روی N/TP tokens کار می‌کند
- نیاز به Ring All-Reduce برای attention

**3D Parallelism — ترکیب همه:**

برای مدل ۱۳۰B روی ۱۰۰۰ GPU، معمولاً ترکیب می‌کنیم:

```
TP=8 (درون node، NVLink سریع)
PP=16 (بین node ها)
DP=8 (data parallelism برای throughput)
Total = 8 × 16 × 8 = 1024 GPU
```

هر data parallel rank یک مدل کامل دارد که با TP+PP distribute شده. ZeRO-1 یا ZeRO-2 برای optimizer states.

**Communication Topology:**

- **NVLink** (۶۰۰ GB/s): برای TP درون node — ضروری است
- **InfiniBand** (۴۰۰ Gb/s): برای PP و DP بین node ها
- **Ethernet** (۱۰۰ Gb/s): برای large DP با ZeRO-3

**Mixed Precision Training:**

- **FP16/BF16** برای activations و forward pass
- **FP32** برای optimizer state و master weights (برای stability)
- **BF16 > FP16**: محدوده dynamic بیشتر، نیازی به loss scaling ندارد

```python
# با PyTorch FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    ),
)
```

**Checkpointing و Fault Tolerance:**

با ۱۰۰۰ GPU، احتمال failure در ۲۴ ساعت بسیار بالاست:
- **Gradient Checkpointing**: به جای نگه داشتن تمام activations در memory، آن‌ها را در backward pass دوباره محاسبه می‌کنیم. 30-40% کاهش memory به قیمت 30-40% افزایش compute
- **Async Checkpointing**: checkpoint در background ذخیره می‌شود بدون متوقف کردن training
- **Elastic Training**: توانایی restart از آخرین checkpoint با GPU های کمتر

**Throughput Metrics:**

- **MFU (Model FLOPs Utilization)**: نسبت FLOPs محاسبه‌شده به peak GPU FLOPs. مقدار خوب: ۳۵-۵۵%
- **Hardware Utilization**: GPU utilization (هدف: > 80%)
- **Tokens/second per GPU**: معیار مقایسه efficiency

**ابزارهای رایج:**
- **Megatron-LM**: از NVIDIA، اکثر مدل‌های بزرگ با آن آموزش دیده‌اند
- **DeepSpeed**: از Microsoft، ZeRO implementation بهترین
- **PyTorch FSDP**: native PyTorch، خوب برای TP+DP
- **JAX/TPU**: برای Google TPU ها، XLA compilation

**اشتباهات رایج کاندیداها:**
- فقط در مورد data parallelism صحبت کردن و فراموش کردن tensor و pipeline parallelism.
- نادیده گرفتن communication overhead — TP روی Ethernet بدون NVLink غیرعملی است.
- فراموش کردن Fault Tolerance — در ۱۰۰۰+ GPU، failure نه exception بلکه norm است.

**سوالات follow-up احتمالی:**
- "چطور optimal parallelism configuration را برای یک مدل و cluster مشخص پیدا می‌کنید؟ (auto-tuning)"
- "Flash Attention چیست و چطور memory complexity attention را از O(N²) به O(N) کاهش می‌دهد؟"
- "اگر بودجه GPU محدود است، چه rویکردی برای آموزش مدل بزرگ دارید؟ (Gradient Accumulation، smaller batch)"

---
### سوال ۱۵: Online Learning System — یادگیری در Real-time
**دسته:** Online Learning / Streaming ML
**سطح:** Staff/Principal
**چرا مهمه:** بسیاری از سیستم‌های ML در دنیای واقعی نیاز دارند تا با داده‌های جدید adapt شوند بدون اینکه کل training را از نو شروع کنند.
**زمان پاسخ:** ۳۵ تا ۵۰ دقیقه

**سوال:** برای یک سیستم تشخیص تقلب (Fraud Detection) در یک شرکت پرداخت آنلاین، می‌خواهید مدل در real-time از تراکنش‌های جدید یاد بگیرد. چه معماری Online Learning طراحی می‌کنید و با چه چالش‌هایی روبرو هستید؟

**پاسخ کامل:**

Online Learning یا Continual Learning به مدل‌ها اجازه می‌دهد از داده‌های جدید به صورت incremental یاد بگیرند. در fraud detection این اهمیت ویژه‌ای دارد چون attackers به سرعت رفتار خود را تغییر می‌دهند.

**چرا Online Learning برای Fraud Detection حیاتی است:**

Fraud patterns مداوم تغییر می‌کنند — این به عنوان "Adversarial Concept Drift" شناخته می‌شود. یک مدل batch که هر ماه یک بار train می‌شود، در برابر attack patterns جدید vulnerable است. مدل باید در ساعت‌ها نه ماه‌ها adapt شود.

**معماری کلی:**

```
[Real-time Transactions] → [Kafka] → [Online Learning Engine] → [Updated Model] → [Serving]
              ↓                               ↑
        [Label Service]              [Feedback Loop]
    (Ground Truth: Fraud/Not)
```

**لایه ۱: Stream Processing با Kafka**

هر تراکنش به عنوان event در Kafka publish می‌شود:
```json
{
  "transaction_id": "txn_123",
  "timestamp": "2026-04-18T08:00:00Z",
  "user_id": "usr_456",
  "amount": 1500.00,
  "merchant_id": "mrch_789",
  "location": {"lat": 35.7, "lon": 51.4},
  "device_fingerprint": "fp_abc"
}
```

**لایه ۲: Real-time Feature Computation**

با Flink یا Kafka Streams، feature های لحظه‌ای محاسبه می‌شوند:
- تعداد تراکنش‌های این user در ۱ ساعت گذشته
- متوسط مبلغ تراکنش این user در ۷ روز گذشته
- تعداد location های مختلف در ۲۴ ساعت
- velocity features (سرعت تغییر الگو)

**لایه ۳: Online Learning Algorithm**

*گزینه ۱: Gradient Descent Incremental*

برای neural networks، می‌توان با هر mini-batch (یا حتی هر sample) یک step gradient descent انجام داد:

```python
# Online gradient update
for transaction in stream:
    features = compute_features(transaction)
    prediction = model.predict(features)
    
    # وقتی label رسید (با تاخیر)
    if label_available(transaction):
        loss = compute_loss(prediction, actual_label)
        model.update(loss, learning_rate=1e-5)  # learning rate کوچک برای stability
```

**مشکل Catastrophic Forgetting**: اگر مدل فقط از داده‌های اخیر یاد بگیرد، الگوهای قدیمی‌تر را "فراموش" می‌کند. راه‌حل‌ها:
- **Experience Replay**: ذخیره یک buffer از داده‌های تاریخی و mix کردن با داده‌های جدید
- **Elastic Weight Consolidation (EWC)**: جریمه کردن تغییر وزن‌هایی که برای task های قدیمی مهم بوده‌اند
- **Learning Rate Scheduling**: learning rate بسیار کوچک برای داده‌های جدید

*گزینه ۲: Ensemble of Models (بهترین برای Fraud)*

نگه داشتن چند مدل موازی:
- **Long-term model**: هر هفته با sliding window از ۳ ماه گذشته train می‌شود
- **Short-term model**: هر روز با داده‌های ۷ روز گذشته fine-tune می‌شود
- **Real-time model**: با streaming data آپدیت می‌شود

Final prediction = weighted ensemble:
```
P(fraud) = 0.5 × P_long + 0.3 × P_short + 0.2 × P_realtime
```

*گزینه ۳: Bandit Algorithms*

برای feature selection یا model selection در real-time:
- UCB (Upper Confidence Bound): balance بین exploration و exploitation
- Thompson Sampling: Bayesian approach

**لایه ۴: Delayed Labels و Feedback Loop**

بزرگ‌ترین چالش: label های fraud با تاخیر ۲۴-۷۲ ساعت می‌رسند (وقتی کاربر chargeback می‌کند).

*Strategies:*
- **Semi-supervised learning**: از unlabeled data هم یاد بگیریم
- **Weak supervision**: از proxy signals استفاده کنیم (مثلاً block شدن کارت)
- **Active Learning**: برای تراکنش‌های uncertain، review انسانی بخواهیم و label جمع کنیم

**لایه ۵: Online Evaluation**

باید مداوم بدانیم مدل چقدر خوب است:
- **Prequential Error**: قبل از هر prediction، آن را evaluate کنیم — running average of loss
- **Interleaved Test-Then-Train**: test بر روی هر batch قبل از train
- **Adaptive Windowing (ADWIN)**: تشخیص خودکار drift در performance

**چالش‌های کلیدی:**

*۱. Class Imbalance در Streaming:*
در fraud detection، < 0.1% تراکنش‌ها fraud هستند. در streaming باید dynamically reweight کنیم.

*۲. Feature Staleness:*
اگر feature computation تاخیر داشته باشد، training و serving با feature های مختلف کار می‌کنند.

*۳. Adversarial Adaptation:*
Fraudsters الگوهای خود را تغییر می‌دهند — online learning باید از over-adaptation هم جلوگیری کند.

*۴. Model Versioning:*
با آپدیت‌های مداوم، چطور می‌دانیم کدام version مدل روی یک تراکنش خاص inference کرده؟ این برای audit و debugging ضروری است.

**Pipeline Deploy:**

```
Kafka Topic: transactions
    ↓
Flink Job: Real-time feature engineering
    ↓
Redis: Online feature store (latency < 5ms)
    ↓
Model Server (TorchServe): Inference
    ↓
Kafka Topic: predictions
    ↓
Label Join Service: Match با chargebacks
    ↓
Online Trainer: Model update
    ↓
Model Registry: Version control
    ↓
Canary Deployment: 5% → 20% → 100%
```

**Safety Mechanisms:**

- **Gradient Clipping**: جلوگیری از gradient explosions در online update
- **Learning Rate Warmup**: برای مدل جدید، learning rate به تدریج افزایش می‌یابد
- **Rollback**: اگر online performance افت کرد، به آخرین stable checkpoint برگرد
- **Shadow Mode**: مدل به‌روز شده را موازی اجرا کنید قبل از deploy

**ابزارها:**
- **River**: بهترین library Python برای online learning
- **Vowpal Wabbit**: بسیار سریع، برای large-scale online learning
- **Flink ML**: برای streaming ML در مقیاس بزرگ

**اشتباهات رایج کاندیداها:**
- نادیده گرفتن مشکل delayed labels — در fraud detection این بزرگترین چالش است.
- فراموش کردن catastrophic forgetting.
- نداشتن safety mechanism برای bad model updates — اگر online update خراب شود چطور rollback می‌کنید؟
- گفتن "همه چیز را real-time کنیم" بدون توجه به هزینه و پیچیدگی.

**سوالات follow-up احتمالی:**
- "چطور تضمین می‌کنید که online update ها reproducible هستند؟ (برای audit)"
- "اگر attacker از online learning مدل شما آگاه باشد، چطور می‌تواند آن را exploit کند و چطور جلوگیری می‌کنید؟"
- "Multi-armed bandit در fraud detection — چه use case هایی دارد؟"

---

*پایان سوالات مصاحبه — ۲۰۲۶-۰۴-۱۸*

---
