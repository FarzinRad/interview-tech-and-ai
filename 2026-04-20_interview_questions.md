# سوالات مصاحبه فنی - ۲۰ آوریل ۲۰۲۶

---
## بخش اول: Data Science

---

### سوال ۱: Bias-Variance Tradeoff و مدیریت آن در مدل‌های پیچیده
**دسته:** Machine Learning Fundamentals  
**سطح:** Mid/Senior  
**چرا مهمه:** این مفهوم پایه‌ای‌ترین دوراهی در طراحی مدل‌های ML است و درک عمیق آن نشان‌دهنده بلوغ فنی کاندیداست.  
**زمان پاسخ:** ۱۰ تا ۱۵ دقیقه

**سوال:** Bias-Variance Tradeoff را به‌طور کامل توضیح بده. چطور این tradeoff را در یک مدل production-level تشخیص می‌دهی و مدیریت می‌کنی؟ چه تکنیک‌هایی برای کاهش هر کدام وجود دارد؟

**پاسخ کامل:**

Bias-Variance Tradeoff یکی از بنیادی‌ترین مفاهیم در یادگیری ماشین است که رابطه‌ای اساسی بین خطای مدل و پیچیدگی آن را توصیف می‌کند. برای درک کامل این مفهوم، باید ابتدا هر کدام از این دو مؤلفه را به‌صورت مجزا بررسی کنیم.

**Bias (تعصب)** به تفاوت سیستماتیک بین مقادیر پیش‌بینی‌شده توسط مدل و مقادیر واقعی اشاره دارد. از منظر ریاضی، bias برابر است با:

```
Bias[f̂(x)] = E[f̂(x)] - f(x)
```

که در آن `f̂(x)` پیش‌بینی مدل و `f(x)` تابع واقعی است. مدلی با bias بالا به این معناست که حتی اگر داده‌های بی‌نهایت داشته باشیم، مدل همچنان به‌سمت پاسخ‌های اشتباه bias دارد — این نشانه underfitting است. مدل‌های ساده مثل Linear Regression روی داده‌های غیرخطی، یا Decision Tree با depth محدود، نمونه‌های کلاسیک bias بالا هستند.

**Variance (واریانس)** به میزان حساسیت مدل به نوسانات در داده‌های training اشاره دارد:

```
Variance[f̂(x)] = E[f̂(x)²] - (E[f̂(x)])²
```

مدلی با variance بالا به این معناست که با کوچکترین تغییر در training data، پیش‌بینی‌های مدل به شکل چشمگیری تغییر می‌کنند — این نشانه overfitting است. مدل‌های پیچیده مثل Deep Neural Networks بدون regularization یا Decision Trees بدون pruning، در معرض variance بالا هستند.

**خطای کل (Total Error)** را می‌توان به شکل زیر تجزیه کرد:

```
E[(y - f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²
```

که `σ²` نویز تقلیل‌ناپذیر (irreducible noise) موجود در داده است.

**تشخیص در محیط Production:**

در یک سیستم واقعی، تشخیص این مشکل نیازمند monitoring دقیق است. اگر مدل روی training set عملکرد خوبی دارد اما روی validation/test set ضعیف است، این نشانه variance بالا (overfitting) است. اگر مدل روی هر دو set ضعیف باشد، نشانه bias بالا (underfitting) است. ابزارهایی مثل Learning Curves بسیار مفید هستند — اگر با افزودن داده‌های بیشتر validation error بهبود یابد، احتمالاً variance مشکل است؛ اگر این بهبود حاصل نشود، bias مشکل است.

**تکنیک‌های کاهش Bias:**
- استفاده از مدل‌های پیچیده‌تر (افزایش model capacity)
- Feature engineering بهتر برای capture کردن روابط غیرخطی
- کاهش regularization strength
- Ensemble methods مثل Boosting که به‌صورت تدریجی bias را کاهش می‌دهد (AdaBoost، Gradient Boosting)
- استفاده از Polynomial Features برای مدل‌های خطی

**تکنیک‌های کاهش Variance:**
- Regularization: L1 (Lasso) که feature selection انجام می‌دهد، L2 (Ridge) که وزن‌ها را کوچک نگه می‌دارد، Elastic Net که ترکیب هر دو است
- Dropout در Neural Networks
- Cross-validation برای انتخاب hyperparameter مناسب
- Ensemble methods مثل Bagging و Random Forest که از میانگین‌گیری روی چند مدل برای کاهش variance استفاده می‌کنند
- افزایش حجم training data
- Early stopping در gradient descent

**مثال واقعی از صنعت:**
در یک سیستم recommendation در یک شرکت بزرگ، ممکن است یک مدل Matrix Factorization ساده bias بالایی داشته باشد چون روابط غیرخطی پیچیده بین کاربران و آیتم‌ها را capture نمی‌کند. راه‌حل معمول استفاده از Neural Collaborative Filtering یا Two-Tower Models است که capacity بیشتری دارند. اما این مدل‌های پیچیده‌تر ممکن است variance بالا داشته باشند و نیاز به regularization قوی‌تر، dropout، و داده‌های training بیشتری داشته باشند.

**اشتباهات رایج کاندیداها:**
- تعریف bias-variance فقط به‌صورت کلامی بدون بیان فرمول ریاضی
- فراموش کردن irreducible noise در تجزیه خطا
- تصور اینکه همیشه باید بین bias و variance یکی را انتخاب کرد — در واقع تکنیک‌هایی مثل Gradient Boosting با regularization مناسب می‌توانند هر دو را کاهش دهند
- عدم اشاره به روش‌های عملی تشخیص این مشکل در محیط واقعی

**سوالات Follow-up احتمالی:**
- آیا می‌توان همزمان هم bias و هم variance را کاهش داد؟
- چطور Bagging به کاهش variance کمک می‌کند؟
- در deep learning چرا overparameterized models گاهی هم bias و هم variance پایینی دارند (double descent phenomenon)؟

---

### سوال ۲: A/B Testing — طراحی، آماره و دام‌های رایج
**دسته:** Statistical Inference & Experimentation  
**سطح:** Senior  
**چرا مهمه:** A/B testing ستون فقرات تصمیم‌گیری داده‌محور در شرکت‌های بزرگ است. اشتباه در طراحی یا تفسیر آزمایش می‌تواند میلیون‌ها دلار هزینه داشته باشد.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** فرض کن product manager از تو می‌خواهد یک A/B test برای ارزیابی یک feature جدید طراحی کنی. از ابتدا تا انتها این فرایند را توضیح بده. چه اشتباهاتی در A/B testing رایج است و چطور از آن‌ها جلوگیری می‌کنی؟

**پاسخ کامل:**

طراحی یک A/B test صحیح یکی از مهارت‌های محوری یک Data Scientist در یک شرکت product-driven است. این فرایند چندین مرحله مجزا دارد که هر کدام اگر اشتباه انجام شوند می‌توانند نتایج کاملاً گمراه‌کننده‌ای تولید کنند.

**مرحله اول: تعریف hypothesis و متریک**

قبل از هر چیز، باید یک null hypothesis و یک alternative hypothesis روشن داشته باشیم. برای مثال، فرض کن یک دکمه «افزودن به سبد خرید» را از آبی به نارنجی تغییر می‌دهیم:
- H₀: تغییر رنگ دکمه تأثیری بر نرخ تبدیل (conversion rate) ندارد
- H₁: تغییر رنگ دکمه نرخ تبدیل را تغییر می‌دهد (دوطرفه) یا افزایش می‌دهد (یک‌طرفه)

در اینجا باید تصمیم بگیریم آیا از یک‌طرفه یا دوطرفه استفاده کنیم. آزمون یک‌طرفه قدرت آماری بیشتری دارد اما باید قبل از دیدن داده تصمیم گرفته شود. اگر hypothesis ما دارای جهت مشخص است، یک‌طرفه مناسب‌تر است.

**مرحله دوم: تعیین sample size**

این مرحله بسیار مهم و اغلب نادیده گرفته می‌شود. باید سه پارامتر را مشخص کنیم:

۱. **سطح معنی‌داری (α):** معمولاً ۰.۰۵ انتخاب می‌شود، یعنی ۵٪ احتمال خطای نوع اول (false positive)
۲. **توان آزمون (Power = 1 - β):** معمولاً ۸۰٪ یا ۹۰٪، یعنی احتمال تشخیص اثر واقعی اگر وجود داشته باشد
۳. **Minimum Detectable Effect (MDE):** کمترین تفاوتی که از نظر تجاری اهمیت دارد

فرمول تقریبی sample size برای آزمون نسبت‌ها:

```
n ≈ (z_{α/2} + z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²
```

اگر conversion rate پایه ۱۰٪ است و می‌خواهیم افزایش ۱٪ (MDE = 10% relative) را با α=0.05 و power=0.8 تشخیص دهیم، به حدود ۱۵,۰۰۰ نفر در هر گروه نیاز داریم.

**مرحله سوم: راندومیزاسیون**

کاربران باید به‌صورت کاملاً تصادفی به گروه control و treatment تقسیم شوند. معمول‌ترین روش hash-based assignment است: `hash(user_id + experiment_id) % 100`. این روش تضمین می‌کند که:
- یک کاربر همیشه در یک گروه مشخص قرار می‌گیرد (consistency)
- توزیع تصادفی یکنواخت است

**مرحله چهارم: اجرا و monitoring**

در طول آزمایش نباید به نتایج نگاه کرد! این مشکل را **peeking problem** می‌نامند. اگر هر بار که p-value به زیر ۰.۰۵ می‌رسد آزمایش را متوقف کنیم، نرخ false positive واقعی می‌تواند به ۳۰-۴۰٪ برسد. راه‌حل‌ها:
- Sequential testing با alpha spending functions مثل روش O'Brien-Fleming
- Bayesian approaches که inherently مشکل peeking ندارند
- از پیش تعیین کردن مدت آزمایش و رعایت آن

**مرحله پنجم: تحلیل نتایج**

پس از پایان آزمایش، t-test یا z-test برای مقایسه میانگین‌ها انجام می‌دهیم. اما علاوه بر p-value، باید **confidence interval** و **effect size** هم گزارش شود. یک p-value معنی‌دار لزوماً به معنی اهمیت تجاری نیست.

**دام‌های رایج A/B Testing:**

**۱. Multiple Testing Problem (مشکل مقایسه چندگانه):** اگر ۲۰ متریک مختلف را بررسی کنیم، به احتمال زیاد حداقل یکی false positive می‌شود. راه‌حل: Bonferroni correction یا تعریف متریک اصلی از پیش.

**۲. Simpson's Paradox:** ممکن است در کل population یک group بهتر باشد اما در هر زیرگروه بدتر باشد. مثلاً اگر گروه treatment بیشتر از موبایل استفاده کنند و موبایل conversion rate پایین‌تری داشته باشد، نتایج کلی گمراه‌کننده می‌شود.

**۳. Network Effects (Interference):** در شبکه‌های اجتماعی، کاربران گروه control و treatment با هم تعامل دارند. این تداخل می‌تواند نتایج را مخدوش کند. راه‌حل: Cluster-based randomization (randomize by geographic region یا user cluster).

**۴. Novelty Effect:** کاربران ممکن است در ابتدا به‌دلیل تازگی یک feature بیشتر با آن تعامل داشته باشند. آزمایش باید به اندازه کافی طولانی باشد تا این اثر خاموش شود.

**۵. Selection Bias:** اگر راندومیزاسیون صحیح نباشد، گروه‌ها از ابتدا متفاوت هستند. قبل از شروع آزمایش باید یک **A/A test** انجام داد تا صحت راندومیزاسیون تایید شود.

**مثال واقعی:**
در یک پلتفرم streaming مثل Netflix، آزمون A/B برای thumbnail جدید نیازمند در نظر گرفتن network effects نیست (چون کاربران به هم وابسته نیستند) اما باید novelty effect را کنترل کرد. همچنین در کنار click-through rate باید watch time را هم به‌عنوان متریک ثانویه رصد کرد تا مطمئن شویم thumbnail گمراه‌کننده نیست.

**اشتباهات رایج کاندیداها:**
- فراموش کردن محاسبه sample size قبل از آزمایش
- عدم اشاره به peeking problem
- تمرکز فقط بر p-value و نادیده گرفتن practical significance
- عدم ذکر A/A test برای validation

**سوالات Follow-up احتمالی:**
- چطور با آزمایش‌های طولانی‌مدت کنار می‌آیی؟ (long-running experiments)
- اگر نمی‌توانی کاربران را randomize کنی چه می‌کنی؟ (quasi-experiments, difference-in-differences)
- چطور متریک‌های متعارض را اولویت‌بندی می‌کنی؟

---

### سوال ۳: Feature Engineering برای داده‌های Time Series
**دسته:** Feature Engineering & Time Series  
**سطح:** Senior  
**چرا مهمه:** بخش بزرگی از داده‌های دنیای واقعی time series هستند و مهندسی ویژگی مناسب برای آن‌ها تفاوت بین یک مدل خوب و یک مدل عالی است.  
**زمان پاسخ:** ۱۲ تا ۱۸ دقیقه

**سوال:** یک مدل پیش‌بینی فروش ماهانه برای یک retailer بزرگ می‌سازی. چه feature‌هایی از داده‌های raw time series استخراج می‌کنی؟ چه چالش‌هایی در این نوع داده وجود دارد و چطور با آن‌ها مقابله می‌کنی؟

**پاسخ کامل:**

Feature engineering برای Time Series از پیچیده‌ترین چالش‌های Data Science است چون علاوه بر ویژگی‌های معمول، باید ساختار زمانی داده را هم حفظ کرد و از data leakage جلوگیری نمود.

**دسته‌بندی اصلی Feature‌های Time Series:**

**۱. Lag Features (ویژگی‌های تأخیری):**
ساده‌ترین و اغلب قدرتمندترین ویژگی‌ها، مقادیر گذشته متغیر هدف هستند:
- `sales_lag_1`: فروش ماه قبل
- `sales_lag_12`: فروش همین ماه سال گذشته (این برای داده‌های ماهانه بسیار مهم است)
- `sales_lag_3`, `sales_lag_6`: فروش ۳ و ۶ ماه قبل

نکته مهم: باید از ایجاد lag features در training time به شکلی که اطلاعات آینده leak شود جلوگیری کرد. در cross-validation باید از Time Series Split استفاده کرد نه k-fold معمولی.

**۲. Rolling Statistics (آمار پنجره‌ای):**
میانگین، انحراف معیار، min، max در پنجره‌های زمانی مختلف:
- `sales_rolling_mean_3`: میانگین ۳ ماه اخیر
- `sales_rolling_std_6`: انحراف معیار ۶ ماه اخیر (نشان‌دهنده volatility)
- `sales_rolling_max_12`: بیشترین فروش در سال گذشته

این ویژگی‌ها باید با پنجره‌های shift شده محاسبه شوند تا از data leakage جلوگیری شود.

**۳. Calendar Features (ویژگی‌های تقویمی):**
- ماه سال (با one-hot encoding یا cyclical encoding با sin/cos)
- روز هفته
- تعطیلات رسمی (باید به‌صورت دستی یا از API تقویم اضافه شوند)
- فاصله تا تعطیلات مهم (مثل Black Friday، کریسمس)
- آیا ماه اول فصل است؟ (اثرات seasonal)

برای ویژگی‌های دوره‌ای مثل ماه، به‌جای استفاده از مقادیر ۱ تا ۱۲ که فاصله عددی نادرست دارند، از cyclical encoding استفاده کنید:
```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**۴. Trend Features:**
- Linear trend: عدد ماه (نمایانگر روند کلی رشد)
- Log-transformed trend برای رشد exponential
- Detrended series با Hodrick-Prescott filter یا STL decomposition

**۵. External Variables (متغیرهای خارجی):**
- قیمت‌های رقبا
- شاخص‌های اقتصادی (تورم، نرخ بیکاری)
- داده‌های آب‌وهوایی (برای برخی محصولات)
- هزینه تبلیغات
- نرخ ارز (برای کالاهای وارداتی)

**چالش‌های اصلی Time Series:**

**۱. Stationarity (ایستایی):**
اکثر مدل‌های آماری کلاسیک مثل ARIMA فرض می‌کنند که سری زمانی stationary است، یعنی میانگین و واریانس آن در طول زمان ثابت هستند. می‌توان با Augmented Dickey-Fuller test آن را بررسی کرد. برای stationary کردن:
- Differencing: `y_t - y_{t-1}`
- Log transformation برای stabilize کردن واریانس
- Seasonal differencing: `y_t - y_{t-12}`

**۲. Seasonality (فصلی‌بودن):**
داده‌های فروش معمولاً seasonality چندگانه دارند (هفتگی، ماهانه، سالانه). STL decomposition می‌تواند این اجزا را از هم جدا کند:
```
Y_t = Trend_t + Seasonal_t + Residual_t
```

**۳. Data Leakage در Cross-Validation:**
استفاده از k-fold عادی روی time series کاملاً اشتباه است چون داده‌های آینده وارد training می‌شوند. باید از **TimeSeriesSplit** استفاده کرد که همیشه روی داده‌های پس از training set validation انجام می‌دهد.

**۴. Irregularity و Missing Values:**
در داده‌های واقعی، ممکن است برخی روزها/ماه‌ها داده نداشته باشیم. روش‌های imputation مناسب برای time series:
- Forward fill (استفاده از آخرین مقدار معتبر)
- Interpolation خطی یا spline
- مدل‌های خاص time series برای imputation

**مثال واقعی:**
در پروژه پیش‌بینی فروش Walmart، یکی از مهم‌ترین ویژگی‌ها این بود که آیا هفته جاری با Black Friday همپوشانی دارد یا نه. همچنین temperature در هر منطقه برای پیش‌بینی فروش beverages و ice cream اهمیت زیادی داشت.

**اشتباهات رایج کاندیداها:**
- استفاده از k-fold عادی به‌جای TimeSeriesSplit
- ایجاد rolling features بدون shift مناسب (data leakage)
- نادیده گرفتن ویژگی‌های دوره‌ای sin/cos
- فراموش کردن external variables

**سوالات Follow-up احتمالی:**
- چطور با multiple seasonality کنار می‌آیی؟
- Prophet در مقابل ARIMA در چه شرایطی برتری دارد؟
- چطور anomalies در time series را تشخیص می‌دهی؟

---

### سوال ۴: Handling Imbalanced Data در مسائل Classification
**دسته:** Data Preprocessing & Class Imbalance  
**سطح:** Mid/Senior  
**چرا مهمه:** در دنیای واقعی، اکثر مسائل classification با imbalanced data روبرو هستند — تشخیص تقلب، تشخیص بیماری، churn prediction — و handling نادرست این مشکل می‌تواند به مدل‌های کاملاً بی‌فایده منجر شود.  
**زمان پاسخ:** ۱۲ تا ۱۵ دقیقه

**سوال:** یک مدل تشخیص تقلب می‌سازی که در آن فقط ۰.۱٪ تراکنش‌ها تقلب هستند. چطور با این imbalance کنار می‌آیی؟ چه متریک‌هایی استفاده می‌کنی و چرا Accuracy معیار مناسبی نیست؟

**پاسخ کامل:**

Imbalanced data یکی از چالش‌های بسیار رایج در Data Science است، به‌خصوص در حوزه‌هایی مثل fraud detection، پزشکی، امنیت سایبری و churn prediction. درک درست این مشکل و رویکردهای مقابله با آن نشانه‌ای از بلوغ تکنیکی یک data scientist است.

**چرا Accuracy معیار نامناسبی است؟**

تصور کنید ۱۰۰۰ تراکنش داریم که ۹۹۹ تای آن‌ها عادی و ۱ تای آن تقلب است. یک مدل «احمق» که همه تراکنش‌ها را عادی پیش‌بینی می‌کند، accuracy برابر ۹۹.۹٪ خواهد داشت! این مدل کاملاً بی‌فایده است اما از نظر accuracy عالی به نظر می‌رسد. این پدیده را **accuracy paradox** می‌نامند.

**متریک‌های مناسب برای Imbalanced Data:**

**۱. Precision و Recall:**
- **Precision** = TP / (TP + FP): از کل مواردی که مدل تقلب تشخیص داد، چند درصد واقعاً تقلب بودند؟
- **Recall (Sensitivity)** = TP / (TP + FN): از کل تقلب‌های واقعی، چند درصد را مدل تشخیص داد؟

در fraud detection، معمولاً Recall مهم‌تر است چون miss کردن یک تقلب (false negative) هزینه بیشتری دارد تا false positive.

**۲. F1-Score:** میانگین هارمونیک Precision و Recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**۳. F-beta Score:** اگر بخواهیم Recall را بیشتر وزن دهیم (β > 1):
```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

**۴. AUC-ROC:** مساحت زیر منحنی ROC که نشان می‌دهد مدل چقدر می‌تواند بین کلاس‌ها تمایز ایجاد کند. مستقل از threshold است.

**۵. AUC-PR (Precision-Recall Curve):** برای imbalanced data اغلب معتبرتر از AUC-ROC است، چون تأثیر majority class را کم می‌کند.

**تکنیک‌های مقابله با Imbalanced Data:**

**سطح داده:**

**Oversampling - SMOTE (Synthetic Minority Over-sampling Technique):**
SMOTE به‌جای تکرار ساده نمونه‌های minority class، نمونه‌های synthetic جدید ایجاد می‌کند. الگوریتم برای هر نمونه minority، k-nearest neighbor آن را پیدا کرده و یک نمونه جدید در مسیر بین دو نمونه ایجاد می‌کند:
```
x_new = x_i + λ × (x_knn - x_i)    λ ~ Uniform(0, 1)
```
نسخه‌های بهبود یافته مثل ADASYN و Borderline-SMOTE نمونه‌های بیشتری در مناطق مرزی و سخت‌تر ایجاد می‌کنند.

**Undersampling:**
کاهش تعداد نمونه‌های majority class. روش‌های پیشرفته‌تر مثل Tomek Links نمونه‌های مرزی majority class را حذف می‌کنند. معمول است که oversampling و undersampling را با هم ترکیب کنیم.

**سطح الگوریتم:**

**Class Weights:**
اکثر framework‌های ML مثل sklearn از پارامتر `class_weight='balanced'` پشتیبانی می‌کنند. این کار cost را برای اشتباه در minority class بالاتر می‌برد:
```python
model = LogisticRegression(class_weight='balanced')
# یا
model = RandomForestClassifier(class_weight={0: 1, 1: 999})
```

**سطح پس‌پردازش:**

**Threshold Tuning:**
به‌جای استفاده از threshold پیش‌فرض ۰.۵، می‌توان threshold را بهینه کرد. با رسم Precision-Recall curve، threshold‌ای را انتخاب می‌کنیم که بهترین توازن برای نیازهای تجاری را داشته باشد.

**روش‌های Ensemble:**

**BalancedBaggingClassifier:** هر base classifier روی زیرمجموعه‌ای با balance مناسب train می‌شود.

**EasyEnsemble:** چندین classifier روی subsets مختلف undersampled majority class train می‌شوند.

**اشتباه رایج: Sampling قبل از Cross-Validation**

یکی از بزرگ‌ترین اشتباهات، اعمال SMOTE روی کل dataset قبل از cross-validation است. این کار باعث data leakage می‌شود چون نمونه‌های synthetic از روی کل داده ساخته می‌شوند. باید SMOTE فقط روی training fold اعمال شود:

```python
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])
cross_val_score(pipeline, X, y, cv=StratifiedKFold())
```

**مثال واقعی:**
در سیستم‌های fraud detection بانکی، معمولاً از combination استفاده می‌شود: SMOTE برای minority class، class_weight برای مدل، و threshold tuning بر اساس cost matrix که هزینه false negative (missed fraud) و false positive (blocked legitimate transaction) را مشخص می‌کند. هزینه miss کردن یک تراکنش تقلبی ممکن است ۱۰۰ برابر هزینه block کردن یک تراکنش مجاز باشد.

**اشتباهات رایج کاندیداها:**
- پیشنهاد فقط oversampling بدون ذکر خطر data leakage
- فراموش کردن Threshold Tuning به‌عنوان یکی از ساده‌ترین و موثرترین روش‌ها
- عدم توضیح چرا AUC-PR برتر از AUC-ROC است
- نادیده گرفتن cost matrix و منطق کسب‌وکار

**سوالات Follow-up احتمالی:**
- در چه حالتی oversampling به جای undersampling ترجیح می‌دهی؟
- چطور یک cost matrix طراحی می‌کنی؟
- Anomaly Detection به‌عنوان جایگزین classification چه مزایا و معایبی دارد؟

---

### سوال ۵: Causal Inference — فراتر از Correlation
**دسته:** Causal Inference & Statistics  
**سطح:** Staff/Senior  
**چرا مهمه:** شرکت‌های بزرگ مثل Google، Meta و Netflix به شدت روی causal inference سرمایه‌گذاری می‌کنند. تفاوت بین correlation و causation می‌تواند تفاوت بین یک تصمیم خوب و یک تصمیم فاجعه‌بار باشد.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** می‌خواهیم بفهمیم آیا ارسال notification به کاربران باعث افزایش engagement می‌شود. ما داریم که کاربرانی که notification دریافت کردند ۳۰٪ engagement بیشتری دارند. آیا می‌توان نتیجه گرفت که notification باعث این افزایش می‌شود؟ چطور causal effect واقعی را تخمین می‌زنی؟

**پاسخ کامل:**

این سوال قلب مسئله causal inference را لمس می‌کند. پاسخ کوتاه این است که **نه** — نمی‌توان از همبستگی مشاهده‌شده نتیجه علّی گرفت. این مشکل در آمار به **Confounding** معروف است.

**چرا همبستگی دلیل علیت نیست؟**

در مثال بالا، کاربرانی که notification فعال دارند احتمالاً اصولاً engaged‌تر هستند — آن‌ها app را بیشتر استفاده می‌کنند، به notification‌ها اهمیت می‌دهند، و از app راضی‌تر هستند. این ویژگی ذاتی (engagement baseline بالاتر) هم باعث می‌شود notification فعال کنند و هم باعث می‌شود engagement بیشتری داشته باشند. این **Confounder** است.

**چارچوب رسمی: Potential Outcomes Framework (Rubin Causal Model)**

برای هر کاربر i، دو potential outcome داریم:
- `Y_i(1)`: engagement اگر notification دریافت کند
- `Y_i(0)`: engagement اگر notification دریافت نکند

**Causal Effect** برای هر فرد: `τ_i = Y_i(1) - Y_i(0)`

مشکل اساسی این است که هیچ‌وقت نمی‌توانیم هر دو را مشاهده کنیم — این را **Fundamental Problem of Causal Inference** می‌نامند. هدف تخمین **Average Treatment Effect (ATE)** است:
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

**روش‌های تخمین Causal Effect:**

**۱. Randomized Controlled Trial (RCT) — طلای معیار:**
بهترین روش A/B test است که در آن کاربران به‌صورت تصادفی به دو گروه تقسیم می‌شوند. Randomization تضمین می‌کند که confounders بین گروه‌ها توزیع مساوی دارند، پس:
```
ATE = E[Y|Treatment=1] - E[Y|Treatment=0]
```

**۲. Instrumental Variables (IV):**
زمانی که RCT امکان‌پذیر نیست، از یک **instrumental variable (Z)** استفاده می‌کنیم که:
- با treatment (دریافت notification) همبسته است
- تأثیر مستقیمی بر outcome (engagement) ندارد
- با confounders ارتباطی ندارد

مثال: اگر یک مشکل فنی باعث شد برخی کاربران به‌صورت تصادفی notification دریافت نکنند، این یک IV طبیعی است.

**۳. Propensity Score Matching:**
اگر داده‌های observational داریم، می‌توان برای هر کاربر treated، یک کاربر untreated مشابه پیدا کنیم. **Propensity score** احتمال دریافت treatment با توجه به covariates است:
```
e(x) = P(Treatment = 1 | X = x)
```
با logistic regression این score را تخمین می‌زنیم و سپس matching انجام می‌دهیم. این روش تأثیر confounders را کاهش می‌دهد اما فقط confounders اندازه‌گیری شده را کنترل می‌کند.

**۴. Difference-in-Differences (DiD):**
اگر داده‌های panel داریم (قبل و بعد از intervention، برای گروه‌های treated و untreated)، می‌توان از DiD استفاده کرد:
```
DiD = (Ȳ_treated,after - Ȳ_treated,before) - (Ȳ_control,after - Ȳ_control,before)
```
فرض اصلی این روش **parallel trends** است: در غیاب treatment، هر دو گروه روند یکسانی می‌داشتند.

**۵. Regression Discontinuity Design (RDD):**
اگر یک threshold وجود دارد که تعیین می‌کند چه کسی treatment دریافت می‌کند، می‌توان از مقایسه افراد نزدیک به این threshold استفاده کرد. مثلاً اگر notification فقط به کاربرانی که بیش از ۷ روز inactive بودند ارسال می‌شود، مقایسه کاربران ۶ روز vs. ۸ روز inactive یک شبه‌آزمایش طبیعی ایجاد می‌کند.

**Causal Graph (DAG) و Backdoor Criterion:**

رویکرد Pearl از Directed Acyclic Graphs (DAG) برای نمایش روابط علّی استفاده می‌کند. برای تخمین causal effect بدون bias، باید تمام **backdoor paths** را ببندیم — یعنی مسیرهایی که از Treatment به Outcome از طریق confounders می‌روند. **Backdoor criterion** مشخص می‌کند کدام مجموعه از متغیرها را باید کنترل کنیم.

**مثال واقعی:**
در LinkedIn، برای اندازه‌گیری اثر نمایش «افراد ممکن است بشناسید»، نمی‌توان به سادگی کاربرانی که این feature را می‌بینند با کسانی که نمی‌بینند مقایسه کرد. کاربران بیشتر engaged این feature را بیشتر می‌بینند (confounder). راه‌حل: randomized holdout experiment که به‌صورت تصادفی برخی کاربران این feature را نمی‌بینند.

**اشتباهات رایج کاندیداها:**
- قبول ساده‌انگارانه همبستگی به‌عنوان علیت
- ندانستن تفاوت بین observational study و RCT
- فراموش کردن unobserved confounders در propensity score matching
- عدم اشاره به parallel trends assumption در DiD

**سوالات Follow-up احتمالی:**
- چطور تأثیر یک intervention را وقتی نمی‌توان RCT اجرا کرد تخمین می‌زنی؟
- Uplift modeling چیست و چطور با causal inference مرتبط است؟
- چطور validity یک IV را بررسی می‌کنی؟

---
## بخش دوم: AI Engineering
---

### سوال ۶: RAG (Retrieval-Augmented Generation) — طراحی و بهینه‌سازی
**دسته:** LLM Systems & Architecture  
**سطح:** Senior  
**چرا مهمه:** RAG یکی از معماری‌های محوری در سیستم‌های AI تولیدی است و مصاحبه‌کنندگان در FAANG انتظار دارند کاندیدا بتواند یک سیستم RAG production-ready طراحی کند.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** یک سیستم Q&A بر اساس اسناد داخلی شرکت می‌سازی که باید به سوالات کاربران با دقت بالا پاسخ دهد. معماری RAG را از ابتدا طراحی کن. چه چالش‌هایی وجود دارد و چطور کیفیت retrieval و generation را بهبود می‌دهی؟

**پاسخ کامل:**

RAG (Retrieval-Augmented Generation) رویکردی است که knowledge پایه‌ای LLM را با قابلیت بازیابی اطلاعات از منابع خارجی ترکیب می‌کند. این معماری مشکل اصلی LLMها — محدودیت knowledge به زمان training — را برطرف می‌کند و همچنین امکان ارائه پاسخ‌های مستند و قابل‌بررسی را فراهم می‌کند.

**معماری کلی سیستم RAG:**

یک سیستم RAG از دو پایپ‌لاین اصلی تشکیل می‌شود:

**Offline Pipeline (Indexing):**
۱. **Document Loading:** بارگذاری اسناد از منابع مختلف (PDF، Word، Confluence، Notion و...)
۲. **Chunking:** تقسیم اسناد به قطعات کوچک‌تر
۳. **Embedding:** تبدیل هر chunk به vector با استفاده از embedding model
۴. **Indexing:** ذخیره vectorها در یک vector database

**Online Pipeline (Query):**
۱. **Query Understanding:** پردازش سوال کاربر
۲. **Retrieval:** پیدا کردن مرتبط‌ترین chunks
۳. **Reranking:** بهبود ترتیب نتایج
۴. **Generation:** تولید پاسخ با LLM با استفاده از context بازیابی شده

**Chunking Strategy — یکی از مهم‌ترین تصمیمات:**

انتخاب استراتژی chunking تأثیر مستقیم بر کیفیت retrieval دارد:

- **Fixed-size chunking:** ساده‌ترین روش اما می‌تواند context را در وسط قطع کند
- **Semantic chunking:** تقسیم بر اساس مرزهای معنایی (پاراگراف، بخش)
- **Hierarchical chunking:** نگه‌داشتن chunks در سطوح مختلف (paragraph → section → document)
- **Sliding window:** chunks با overlap برای حفظ context در مرزها

بهترین chunk size نیازمند آزمایش است. معمولاً ۵۱۲-۱۰۲۴ توکن با ۲۰٪ overlap نقطه شروع خوبی است.

**Embedding Model Selection:**

نه هر embedding model برای هر use case مناسب است:
- **text-embedding-3-large (OpenAI):** کیفیت بالا، اما API call لازم دارد
- **E5-mistral-7b-instruct:** open-source با کیفیت رقابتی
- **Cohere Embed v3:** بهینه‌شده برای retrieval، پشتیبانی از multilingual
- **BGE-M3:** multi-lingual، multi-granularity — برای اسناد فارسی گزینه خوبی است

**Vector Database:**
- **Pinecone:** managed service، آسان برای شروع
- **Weaviate:** open-source، امکانات hybrid search
- **Qdrant:** open-source، سریع و کارآمد
- **pgvector:** اگر از PostgreSQL استفاده می‌کنید، نیازی به DB جداگانه نیست

**بهبود کیفیت Retrieval:**

**Hybrid Search:**
ترکیب sparse retrieval (BM25/keyword) با dense retrieval (vector similarity) معمولاً از هر کدام به‌تنهایی بهتر است. برای ترکیب نتایج از **Reciprocal Rank Fusion (RRF)** استفاده می‌شود:
```
RRF_score(doc) = Σ 1/(k + rank_i(doc))    k=60 (constant)
```

**Query Transformation:**
- **HyDE (Hypothetical Document Embeddings):** ابتدا با LLM یک پاسخ فرضی generate می‌کنیم، سپس embedding آن پاسخ را برای retrieval استفاده می‌کنیم
- **Query Expansion:** با LLM چند paraphrase از سوال اصلی generate می‌کنیم
- **Step-back prompting:** سوال را به یک سوال کلی‌تر تبدیل می‌کنیم

**Reranking:**
بعد از retrieval اولیه، یک cross-encoder reranker برای ranking دقیق‌تر استفاده می‌شود. مدل‌هایی مثل Cohere Reranker یا BGE-Reranker document و query را به‌صورت جفت می‌بینند (نه embedding جداگانه) و scoring دقیق‌تری انجام می‌دهند.

**Evaluation Framework:**

برای ارزیابی سیستم RAG از چارچوب **RAGAS** استفاده می‌شود:
- **Faithfulness:** آیا پاسخ فقط از context بازیابی شده استفاده می‌کند؟
- **Answer Relevancy:** آیا پاسخ به سوال مرتبط است؟
- **Context Precision:** آیا chunks بازیابی شده مرتبط هستند؟
- **Context Recall:** آیا همه اطلاعات لازم بازیابی شدند؟

**چالش‌های پیشرفته:**

**۱. Multi-hop Reasoning:** سوالاتی که نیازمند ترکیب اطلاعات از چند سند هستند. راه‌حل: iterative retrieval که در چند step انجام می‌شود.

**۲. Long Context:** مدل‌های LLM با contexts طولانی دچار «Lost in the Middle» می‌شوند — اطلاعات وسط context اغلب نادیده گرفته می‌شود. راه‌حل: reranking و قرار دادن مهم‌ترین chunks در ابتدا و انتهای context.

**۳. Metadata Filtering:** استفاده از metadata (تاریخ سند، نویسنده، دپارتمان) برای pre-filtering قبل از vector search.

**مثال واقعی:**
Notion AI از RAG برای پاسخ به سوالات درباره محتوای workspace استفاده می‌کند. آن‌ها با ترکیب hierarchical chunking، hybrid search، و reranking توانستند دقت پاسخ‌ها را به‌طور قابل توجهی افزایش دهند.

**اشتباهات رایج کاندیداها:**
- طراحی سیستم بدون در نظر گرفتن evaluation
- نادیده گرفتن chunking strategy و فرض اینکه فقط embedding model مهم است
- عدم اشاره به hybrid search
- فراموش کردن latency requirements در production

**سوالات Follow-up احتمالی:**
- چطور با اسناد multilingual کنار می‌آیی؟
- GraphRAG در مقابل RAG سنتی چه مزایایی دارد؟
- چطور freshness داده در vector database را مدیریت می‌کنی؟

---

### سوال ۷: Fine-tuning در مقابل Prompt Engineering — چه زمانی کدام را انتخاب کنیم؟
**دسته:** LLM Adaptation & Optimization  
**سطح:** Senior/Staff  
**چرا مهمه:** این یکی از رایج‌ترین سوالات در مصاحبه‌های AI Engineering است. انتخاب اشتباه می‌تواند هزاران دلار هزینه و ماه‌ها زمان هدر دهد.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** یک شرکت می‌خواهد یک LLM را برای یک task خاص بهینه کند. چه زمانی Prompt Engineering کافی است و چه زمانی به Fine-tuning نیاز است؟ انواع fine-tuning را توضیح بده و trade-off‌های هر رویکرد را بیان کن.

**پاسخ کامل:**

این تصمیم یکی از مهم‌ترین architectural decisions در پروژه‌های AI Engineering است. قانون کلی این است که **همیشه با ساده‌ترین رویکرد شروع کنید** — اگر prompt engineering جواب می‌دهد، نیازی به fine-tuning نیست.

**Prompt Engineering — کی کافی است؟**

Prompt Engineering زمانی کافی است که:
- Task نیازمند دانش جدیدی که در مدل نیست نباشد
- مدل پایه قابلیت‌های لازم را دارد اما نیازمند راهنمایی است
- Training data کافی برای fine-tuning ندارید
- نیاز به سرعت iteration بالا دارید
- Budget محدود است

**تکنیک‌های پیشرفته Prompt Engineering:**
- **Few-shot prompting:** ارائه چند مثال در prompt
- **Chain-of-Thought (CoT):** درخواست از مدل برای توضیح مرحله به مرحله تفکرش
- **ReAct:** ترکیب reasoning و action
- **System prompt optimization:** تعریف دقیق role، output format، و constraints

**چه زمانی Fine-tuning لازم است؟**

۱. **Style و Format خاص:** وقتی می‌خواهید مدل به شکل بسیار خاصی پاسخ دهد که با prompt کنترل آن سخت است
۲. **Domain-specific Knowledge:** وقتی دانش تخصصی که در مدل نیست مورد نیاز است (اسناد داخلی، زبان خاص صنعت)
۳. **Latency و Cost:** مدل fine-tuned کوچک‌تر می‌تواند به عملکرد مدل بزرگ‌تر با prompt طولانی برسد
۴. **Consistency:** وقتی نیاز به رفتار کاملاً یکسان در سناریوهای مشابه دارید
۵. **Prompt Injection Security:** مدل fine-tuned به اندازه کمتری نسبت به دستورالعمل‌های جاسازی‌شده در context حساس است

**انواع Fine-tuning:**

**Full Fine-tuning:**
به‌روز کردن تمام پارامترهای مدل. گران‌قیمت، نیازمند hardware زیاد، و خطر catastrophic forgetting دارد. معمولاً برای small-to-medium models مناسب است.

**Parameter-Efficient Fine-Tuning (PEFT):**

**LoRA (Low-Rank Adaptation):**
به‌جای به‌روز کردن کل weight matrix W، تغییر آن را با حاصل‌ضرب دو ماتریس کم‌رتبه تقریب می‌زنیم:
```
W' = W + ΔW = W + A×B
```
که A ∈ R^(d×r) و B ∈ R^(r×d) و r << d. معمولاً r=4 تا r=64.

**مزایا:** کاهش ۱۰۰۰۰x در پارامترهای trainable، کاهش چشمگیر memory. **معایب:** عملکرد کمی پایین‌تر از full fine-tuning.

**QLoRA (Quantized LoRA):**
LoRA + Quantization مدل پایه به 4-bit. امکان fine-tuning مدل‌های ۷۰B را روی یک A100 80GB فراهم می‌کند.

**Adapters:**
اضافه کردن لایه‌های کوچک «adapter» بین لایه‌های transformer. فقط این لایه‌ها در طول training به‌روز می‌شوند.

**Prefix Tuning / Prompt Tuning:**
اضافه کردن token‌های trainable به ابتدای sequence. سریع‌ترین روش اما کمترین انعطاف را دارد.

**RLHF و RLAIF:**

برای alignment مدل با ارزش‌های انسانی، از Reinforcement Learning از Human Feedback (RLHF) استفاده می‌شود:
۱. Supervised Fine-Tuning (SFT) روی demonstrations با کیفیت
۲. Reward Model Training از ترجیحات انسانی
۳. Policy Optimization با PPO یا DPO

**DPO (Direct Preference Optimization)** جایگزین ساده‌تری برای RLHF است که مستقیماً از داده‌های preference بدون reward model جداگانه استفاده می‌کند.

**Data Requirements:**

یکی از چالش‌های اصلی fine-tuning، نیاز به داده با کیفیت است:
- **SFT:** حداقل ۵۰۰-۱۰۰۰ نمونه با کیفیت بالا (کیفیت از کمیت مهم‌تر است)
- **LoRA:** می‌توان با کمتر از ۱۰۰ نمونه نتایج قابل قبولی گرفت
- **DPO:** نیازمند جفت‌های (chosen, rejected) response

**Evaluation و Alignment Tax:**

بعد از fine-tuning باید مطمئن شویم:
- مدل روی task اصلی بهتر شده
- مدل قابلیت‌های عمومی خود را از دست نداده (catastrophic forgetting)
- مدل رفتارهای مضر را «فراموش» نکرده

**مثال تصمیم‌گیری:**
یک شرکت حقوقی می‌خواهد مدلی بسازد که قراردادها را بررسی کند. رویکرد پیشنهادی:
۱. ابتدا با GPT-4 + few-shot prompt از قراردادهای نمونه آزمایش کنید
۲. اگر نتایج قابل قبول است، از RAG + prompt استفاده کنید (اضافه کردن اسناد قانونی مرتبط به context)
۳. اگر نیاز به style خاص یا اصطلاحات بسیار تخصصی هست، LoRA fine-tuning روی یک مدل ۷B مثل Llama-3

**اشتباهات رایج کاندیداها:**
- پیشنهاد fine-tuning بلافاصله بدون آزمایش prompt engineering
- عدم ذکر LoRA/PEFT و فرض اینکه fine-tuning = full fine-tuning
- نادیده گرفتن data quality در favor of quantity
- فراموش کردن evaluation post fine-tuning

**سوالات Follow-up احتمالی:**
- چطور catastrophic forgetting را کاهش می‌دهی؟
- Continual Learning در context LLM چه معنایی دارد؟
- چطور cost fine-tuning vs. cost در inference را trade-off می‌کنی؟

---

### سوال ۸: LLM Evaluation — چطور کیفیت LLM را اندازه می‌گیریم؟
**دسته:** LLM Evaluation & Metrics  
**سطح:** Senior  
**چرا مهمه:** Evaluation یکی از سخت‌ترین چالش‌های LLM است. بدون evaluation صحیح، نمی‌توان بهبود واقعی را از توهم بهبود تشخیص داد.  
**زمان پاسخ:** ۱۲ تا ۱۸ دقیقه

**سوال:** یک LLM-based application (مثلاً یک coding assistant) را چطور ارزیابی می‌کنی؟ چالش‌های evaluation را توضیح بده و یک framework جامع ارائه ده.

**پاسخ کامل:**

Evaluation یکی از پیچیده‌ترین و اغلب نادیده گرفته‌شده‌ترین بخش‌های LLM applications است. برخلاف ML سنتی که متریک‌های کمی واضحی مثل accuracy یا RMSE دارند، evaluation LLMها نیازمند رویکرد چندوجهی است.

**سطوح مختلف Evaluation:**

**۱. Component-level Evaluation:**
ارزیابی هر component سیستم به‌صورت مجزا. برای coding assistant:
- کیفیت code generation
- دقت bug detection
- کیفیت code explanation

**۲. System-level Evaluation:**
ارزیابی کل pipeline از دید کاربر نهایی. آیا کاربر توانست با استفاده از assistant کار خود را انجام دهد؟

**۳. Safety و Alignment Evaluation:**
- آیا مدل denylist را رعایت می‌کند؟
- آیا hallucination تولید می‌کند؟
- آیا biases مضر دارد؟

**متریک‌های Automatic:**

**برای Text Generation:**
- **BLEU:** مقایسه n-gram بین generated و reference text. برای translation خوب است اما برای open-ended generation ضعیف است
- **ROUGE:** بر اساس recall از reference text. برای summarization مناسب‌تر است
- **BERTScore:** استفاده از BERT embeddings برای مقایسه معنایی — بهتر از BLEU/ROUGE

**برای Code Generation:**
- **pass@k:** احتمال اینکه از k کد تولیدی، حداقل یکی test‌ها را pass کند
- **execution accuracy:** آیا کد اجرا می‌شود؟
- **HumanEval، MBPP، SWE-bench:** benchmark‌های استاندارد

**LLM-as-a-Judge:**

یکی از قدرتمندترین رویکردهای modern، استفاده از یک LLM قوی (مثل GPT-4) به‌عنوان judge است. این LLM کیفیت output را بر اساس criteria از پیش تعریف‌شده score می‌دهد:

```
Criteria مثال برای coding assistant:
- Correctness (0-5): آیا کد منطق درستی دارد؟
- Code Quality (0-5): آیا کد readable و maintainable است؟
- Completeness (0-5): آیا تمام requirements را پوشش داده؟
- Explanation Quality (0-5): آیا توضیح واضح است؟
```

**مزایا:** انعطاف‌پذیر، مقیاس‌پذیر، نزدیک به ارزیابی انسانی
**معایب:** گران، latency بالا، positional bias (LLM اولین پاسخ را ترجیح می‌دهد)، verbosity bias

**Techniques برای کاهش Bias در LLM-as-a-Judge:**
- تعویض ترتیب پاسخ‌ها و میانگین‌گیری
- Multi-aspect evaluation به‌جای single score
- استفاده از چند LLM judge مختلف

**Human Evaluation:**

برای fine-tuning، alignment، و validation نهایی، human evaluation ضروری است:
- **Likert Scale:** کاربران پاسخ را از ۱ تا ۵ رتبه‌بندی می‌کنند
- **Pairwise Comparison (A/B):** به کاربران دو پاسخ نشان داده می‌شود و باید بهتر را انتخاب کنند
- **Task Completion:** اندازه‌گیری اینکه آیا کاربر توانست task را با کمک assistant انجام دهد

**Inter-Annotator Agreement** مثل Cohen's Kappa باید اندازه‌گیری شود تا مطمئن شویم human annotations reliable هستند.

**Evaluation Benchmarks:**

برای مقایسه با baseline‌های عمومی:
- **MMLU:** دانش عمومی در ۵۷ حوزه
- **HumanEval / SWE-bench Verified:** coding
- **TruthfulQA:** hallucination measurement
- **MT-Bench، LMSYS Chatbot Arena:** چندمرحله‌ای و conversation

**Anti-patterns در Evaluation:**

**Overfitting به Benchmark:** اگر مدل مستقیماً روی benchmark data train شود، نتایج گمراه‌کننده است. باید از held-out test sets استفاده کرد.

**تمرکز فقط بر Automatic Metrics:** BLEU بالا به معنی پاسخ خوب نیست. همیشه باید human spot-checking انجام شود.

**عدم توجه به Failure Modes:** باید به‌صورت systematic failure cases را بررسی کرد — نه فقط average performance.

**Red Teaming:** به‌صورت هدفمند سعی می‌کنیم مدل را fail کنیم تا ضعف‌های آن را پیدا کنیم.

**Testing Pyramid برای LLM:**

مثل software testing، یک testing pyramid برای LLM هم مفید است:
- **Unit Tests (پایه):** test‌های خاص برای رفتارهای مشخص (مدل باید به این input پاسخ درست بدهد)
- **Integration Tests (میانه):** ارزیابی workflow کامل
- **End-to-End Tests (بالا):** ارزیابی user journey کامل

**مثال واقعی:**
GitHub Copilot از ترکیب pass@k روی HumanEval، LLM-as-a-Judge برای code quality، و human evaluation از طریق feedback کاربران («این suggestion مفید بود؟») برای ارزیابی سیستم استفاده می‌کند.

**اشتباهات رایج کاندیداها:**
- اکتفا کردن به BLEU/ROUGE که برای LLMs نامناسب‌اند
- نادیده گرفتن safety و alignment evaluation
- عدم اشاره به human evaluation
- فراموش کردن اینکه evaluation باید iterative و continuous باشد

**سوالات Follow-up احتمالی:**
- چطور evaluation data set می‌سازی وقتی ground truth نداری؟
- چطور regression در model updates را تشخیص می‌دهی؟
- Evals برای RAG چه تفاوتی با evals عمومی دارند؟

---

### سوال ۹: LLM Agents — طراحی و چالش‌ها
**دسته:** AI Agents & Orchestration  
**سطح:** Staff  
**چرا مهمه:** Agentic AI یکی از داغ‌ترین حوزه‌های AI Engineering است. شرکت‌های بزرگ به دنبال افرادی هستند که می‌توانند سیستم‌های agent-based طراحی و implement کنند.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** یک AI Agent برای تحلیل مالی طراحی کن که می‌تواند به‌طور خودکار داده‌های بازار را جمع‌آوری کند، تحلیل کند، و گزارش تولید کند. چه چالش‌هایی در طراحی agents وجود دارد و چطور reliability را تضمین می‌کنی؟

**پاسخ کامل:**

LLM Agents سیستم‌هایی هستند که از یک LLM به‌عنوان «مغز» استفاده می‌کنند و قادرند به‌صورت خودکار با محیط تعامل کنند، تصمیم بگیرند، و اقدام کنند. این معماری در حال تغییر اساسی نحوه ساخت نرم‌افزار است.

**اجزای اصلی یک Agent:**

**۱. Perception (درک محیط):**
Agent باید بتواند input‌های مختلف را پردازش کند: متن، داده‌های structured، نتایج API، خروجی ابزارها.

**۲. Memory (حافظه):**
- **In-context memory:** اطلاعات در context window LLM
- **External memory:** Vector database، Key-value store
- **Episodic memory:** تاریخچه مکالمات و اقدامات گذشته
- **Semantic memory:** دانش عمومی و facts

**۳. Planning (برنامه‌ریزی):**
LLM تصمیم می‌گیرد چه اقداماتی باید انجام شود. رویکردهای مختلف:
- **ReAct:** Reasoning + Acting — مدل فکر می‌کند، سپس عمل می‌کند، سپس نتیجه را مشاهده می‌کند
- **Plan and Execute:** ابتدا یک plan کامل می‌سازد، سپس آن را execute می‌کند
- **Tree of Thoughts:** در هر مرحله چند مسیر مختلف بررسی می‌کند و بهترین را انتخاب می‌کند

**۴. Action (اجرا با ابزارها):**
Agent از tools استفاده می‌کند:
- Web search و data retrieval
- Code execution
- API calls
- Database queries
- File operations

**طراحی Financial Analysis Agent:**

```
Financial Analysis Agent
├── Data Collection Tools
│   ├── market_data_api (Yahoo Finance, Bloomberg API)
│   ├── news_scraper (financial news)
│   ├── sec_filing_reader (SEC EDGAR)
│   └── economic_indicators (FRED API)
├── Analysis Tools
│   ├── technical_analysis (MA, RSI, MACD)
│   ├── fundamental_analysis (P/E, DCF)
│   ├── sentiment_analysis (news sentiment)
│   └── code_executor (Python/pandas)
├── Memory
│   ├── vector_db (research history)
│   └── structured_db (historical analysis)
└── Output Tools
    ├── report_generator
    └── chart_creator
```

**Workflow نمونه:**

۱. Agent دستور «تحلیل سهام AAPL انجام بده» را دریافت می‌کند
۲. ReAct: «باید قیمت تاریخی، اخبار اخیر، و گزارش مالی را بررسی کنم»
۳. Tool call: `get_stock_data("AAPL", period="1y")`
۴. Tool call: `search_news("Apple Inc financial news", limit=10)`
۵. Tool call: `run_python_code("calculate_technical_indicators(data)")`
۶. Tool call: `get_sec_filing("AAPL", "10-K")`
۷. Synthesis: ترکیب همه داده‌ها و تولید گزارش

**چالش‌های اصلی Agents:**

**۱. Reliability و Error Handling:**
agents می‌توانند در هر مرحله fail کنند. باید:
- Retry logic با exponential backoff برای API failures
- Fallback strategies (اگر API X fail شد، از API Y استفاده کن)
- Checkpoint saving برای long-running tasks
- Maximum step count برای جلوگیری از infinite loops

**۲. Hallucination در Tool Use:**
LLM ممکن است ابزارهایی که وجود ندارند فراخوانی کند یا آرگومان‌های اشتباه بدهد. راه‌حل:
- Validation دقیق tool schemas
- Structured output با Pydantic validation
- Confirmation steps برای destructive actions

**۳. Context Window Management:**
در tasks طولانی، context window پر می‌شود. راه‌حل:
- Summarization از مراحل قبلی
- External memory برای نگه‌داری اطلاعات مهم
- Selective context — فقط اطلاعات relevant را نگه داشتن

**۴. Human-in-the-Loop:**
برای اقدامات پرریسک (مثل خرید/فروش سهام)، باید confirmation از انسان بگیریم. باید مکانیزمی برای pause کردن agent و دریافت تأیید داشته باشیم.

**۵. Evaluation و Debugging:**
debugging یک agent بسیار سخت‌تر از debugging یک مدل ساده است. باید:
- Comprehensive logging از تمام tool calls و reasoning
- Replay capability برای debug
- Tracing tools مثل LangSmith یا Weights & Biases

**Multi-Agent Systems:**
برای tasks پیچیده‌تر، می‌توان از چند agent تخصصی استفاده کرد:
- **Orchestrator Agent:** تصمیم می‌گیرد کدام sub-agent مناسب است
- **Specialized Agents:** هر کدام متخصص یک حوزه هستند

**Security:**
- Prompt injection: مراقب data بازیابی‌شده که ممکن است دستورالعمل‌های مخرب داشته باشد
- Tool permissions: principle of least privilege
- Sandboxed execution برای code

**اشتباهات رایج کاندیداها:**
- طراحی بدون در نظر گرفتن failure modes
- نادیده گرفتن human-in-the-loop برای high-stakes actions
- فراموش کردن observability و logging
- عدم ذکر context management

**سوالات Follow-up احتمالی:**
- چطور cost per task را در agent systems کاهش می‌دهی؟
- Autonomous vs. supervised agents — چه trade-off‌هایی وجود دارد؟
- چطور با conflicts بین agents در multi-agent system کنار می‌آیی؟

---

### سوال ۱۰: Inference Optimization — LLM را در Production سریع‌تر کن
**دسته:** LLM Serving & Optimization  
**سطح:** Senior/Staff  
**چرا مهمه:** Cost و latency inference یکی از بزرگ‌ترین چالش‌های عملی deployment LLM است. بهینه‌سازی inference می‌تواند ۱۰x کاهش هزینه داشته باشد.  
**زمان پاسخ:** ۱۵ تا ۲۰ دقیقه

**سوال:** یک LLM با ۷ میلیارد پارامتر را باید در production با target latency زیر ۱۰۰ms (Time to First Token) و throughput بالا deploy کنی. چه تکنیک‌هایی برای بهینه‌سازی inference استفاده می‌کنی؟

**پاسخ کامل:**

Inference optimization برای LLM یکی از چالش‌های engineering‌ی است که در آن درک عمیق از معماری transformer، hardware، و system design با هم ترکیب می‌شود.

**درک مشکل اصلی:**

LLM inference از دو مرحله تشکیل می‌شود:
۱. **Prefill:** پردازش تمام input tokens به‌صورت موازی — compute-bound
۲. **Decoding:** تولید output tokens یک به یک — memory-bandwidth-bound

Time to First Token (TTFT) به مرحله prefill و Time per Output Token (TPOT) به مرحله decoding وابسته است.

**تکنیک ۱: KV Cache**

در هر مرحله decoding، attention به تمام tokens قبلی محاسبه می‌شود. بدون caching، این محاسبات تکراری هستند. **KV Cache** key و value vectorهای attention را ذخیره می‌کند تا از محاسبه مجدد جلوگیری شود. این تکنیک جزء built-in معماری استاندارد inference است.

**تکنیک ۲: Quantization**

کاهش دقت پارامترها از FP32 یا BF16 به INT8 یا INT4:

- **INT8 Quantization (W8A8):** وزن‌ها و activations به 8-bit. تقریباً ۲x کاهش memory با افت کیفیت جزئی
- **GPTQ (W4A16):** وزن‌ها 4-bit اما activations در FP16. کاهش ۴x در memory مدل
- **AWQ (Activation-aware Weight Quantization):** smarter از GPTQ، وزن‌های مهم‌تر را با دقت بیشتر نگه می‌دارد

برای مدل ۷B:
- FP16: ~14 GB VRAM
- INT8: ~7 GB VRAM  
- INT4: ~4 GB VRAM

**تکنیک ۳: Continuous Batching**

در serving سنتی، batch تا زمانی که همه sequence‌ها تمام شوند منتظر می‌ماند. **Continuous Batching** (یا dynamic batching) به محض آزاد شدن جای یک sequence، sequence جدیدی را در همان batch اضافه می‌کند. این کار throughput را چندین برابر افزایش می‌دهد.

**تکنیک ۴: Speculative Decoding**

یک مدل کوچک‌تر (draft model) چند token را پیش‌بینی می‌کند، سپس مدل اصلی (target model) همه آن‌ها را به‌صورت موازی verify می‌کند. اگر target model تأیید کند، چند token در یک forward pass تولید می‌شوند. این تکنیک می‌تواند ۲-۳x speedup ایجاد کند.

**تکنیک ۵: Paged Attention**

مدیریت KV Cache مانند virtual memory در OS:
- memory را به «pages» تقسیم می‌کند
- از fragmentation جلوگیری می‌کند
- امکان اشتراک KV Cache بین requests مشابه را فراهم می‌کند (prefix sharing)
- این پایه vLLM است که یکی از محبوب‌ترین serving engines است

**تکنیک ۶: Flash Attention**

بهینه‌سازی سطح GPU kernel برای محاسبه attention:
- داده را tile‌بندی می‌کند تا مناسب SRAM GPU باشد
- از round-trips به HBM جلوگیری می‌کند
- تقریباً ۲-۴x سریع‌تر از attention استاندارد

**Tensor Parallelism و Pipeline Parallelism:**

برای مدل‌های بزرگ‌تر که روی یک GPU جا نمی‌شوند:
- **Tensor Parallelism:** هر layer را روی چند GPU تقسیم می‌کند
- **Pipeline Parallelism:** لایه‌های مختلف روی GPUهای مختلف

**Model Architecture Choices:**

- **GQA (Grouped Query Attention):** تعداد KV heads را کاهش می‌دهد (مثل Llama-2 و GPT-4 Turbo)
- **MoE (Mixture of Experts):** فقط بخشی از پارامترها برای هر token فعال می‌شوند — Mistral، Mixtral

**Serving Frameworks:**

- **vLLM:** bپد PagedAttention، continuous batching، throughput بالا
- **TGI (Text Generation Inference):** از Hugging Face، production-ready
- **TensorRT-LLM:** بهینه‌شده برای NVIDIA GPUs
- **llama.cpp:** برای اجرا روی CPU

**Stack بهینه برای target ما:**

برای ۷B با TTFT < 100ms:
۱. **Quantization:** AWQ 4-bit → 4 GB VRAM (روی A10G کافی است)
۲. **vLLM** با PagedAttention و continuous batching
۳. **Speculative Decoding** با یک مدل 1B draft
۴. **Flash Attention 2** برای prefill سریع‌تر
۵. **Prefix Caching** برای system prompts تکراری

**Benchmarking:**

قبل و بعد از هر optimization باید benchmark کرد:
- TTFT، TPOT، End-to-End latency
- Throughput (tokens/second per GPU)
- GPU Memory Utilization

**اشتباهات رایج کاندیداها:**
- فقط ذکر quantization بدون سایر تکنیک‌ها
- نادیده گرفتن تفاوت prefill vs. decoding bottleneck
- عدم اشاره به continuous batching که اغلب بیشترین تأثیر را دارد
- فراموش کردن benchmarking

**سوالات Follow-up احتمالی:**
- چطور cost per token را محاسبه و بهینه می‌کنی؟
- در چه شرایطی CPU inference برای LLM منطقی است؟
- چطور با spiky traffic در LLM serving کنار می‌آیی؟

---
## بخش سوم: DL/ML System Design
---

### سوال ۱۱: طراحی Recommendation System برای یک پلتفرم Video Streaming
**دسته:** Recommendation Systems  
**سطح:** Senior/Staff  
**چرا مهمه:** Recommendation systems در قلب بسیاری از شرکت‌های بزرگ مثل Netflix، YouTube، و Spotify هستند. طراحی یک سیستم recommendation production-grade یکی از رایج‌ترین سوالات ML System Design است.  
**زمان پاسخ:** ۳۰ تا ۴۵ دقیقه

**سوال:** یک سیستم recommendation برای یک پلتفرم video streaming با ۱۰۰ میلیون کاربر و ۱۰ میلیون video طراحی کن. سیستم باید homepage هر کاربر را شخصی‌سازی کند. Requirements، معماری، و trade-off‌ها را توضیح بده.

**پاسخ کامل:**

این نوع سوال در FAANG interviews بسیار رایج است و انتظار می‌رود کاندیدا مثل یک senior ML engineer فکر کند — نه فقط یک data scientist. باید requirements را clarify کرد، سیستم را به اجزای مختلف تجزیه کرد، و trade-off‌های هر تصمیم را توضیح داد.

**مرحله ۱: Clarify Requirements**

قبل از هر چیز، requirements را روشن کنیم:

**Functional Requirements:**
- کاربر homepage شخصی‌سازی‌شده با ۲۰-۵۰ توصیه دریافت می‌کند
- توصیه‌ها باید بر اساس تاریخچه تماشا، likes، و مشابه‌ها باشد
- جدیدترین ویدئوها باید هم در توصیه‌ها نمایش یابند (freshness)

**Non-functional Requirements:**
- Latency: < ۲۰۰ms برای بارگذاری homepage
- Scale: ۱۰۰M کاربر، ۱۰M ویدئو
- Throughput: اگر ۱٪ کاربران همزمان active باشند: ۱M request/sec
- Training: مدل باید روزانه re-train شود

**مرحله ۲: Data و Feature Engineering**

**User Features:**
- demographics (سن، جنسیت، مکان)
- تاریخچه تماشا (۱۰۰ ویدئو اخیر)
- explicit feedback (likes، ratings، thumbs up/down)
- implicit feedback (completion rate، pause events، rewinds)
- آخرین جستجوها
- device type و time of day

**Item (Video) Features:**
- metadata: عنوان، توضیحات، genre، language، duration
- content embedding: از visual و audio content
- popularity signals: view count، trending score
- freshness: زمان انتشار
- engagement metrics: average completion rate، skip rate

**Interaction Features:**
- Collaborative Filtering embeddings
- Co-watch patterns (ویدئوهایی که معمولاً با هم تماشا می‌شوند)

**مرحله ۳: معماری Two-Stage Pipeline**

سیستم recommendation در مقیاس بزرگ نیازمند معماری دو مرحله‌ای است:

**Stage 1: Candidate Generation (Retrieval)**

از میان ۱۰ میلیون ویدئو، ۱۰۰۰-۵۰۰۰ کاندیدا انتخاب کنیم.

**مدل‌های Retrieval:**

۱. **Collaborative Filtering با Matrix Factorization:**
تجزیه ماتریس تعامل user-item به دو ماتریس با ابعاد پایین‌تر:
```
R ≈ U × V^T
```
برای prediction: `score(u, i) = u_embedding · i_embedding`

۲. **Two-Tower Neural Network:**
دو برج جداگانه برای user و item embedding، output از هر برج یک dense vector است. در train time با dot product و softmax loss. در inference از ANN (Approximate Nearest Neighbor) برای جستجوی سریع استفاده می‌کنیم.

۳. **Graph-based Methods:**
YouTube از Random Walk روی user-item bipartite graph استفاده می‌کند.

**ANN Indexing:**
برای جستجوی سریع در فضای embedding از روش‌هایی مثل FAISS (Facebook AI Similarity Search) استفاده می‌شود که می‌تواند در چند میلی‌ثانیه از میان میلیون‌ها vector nearest neighbors را پیدا کند.

**Stage 2: Ranking**

۱۰۰۰ کاندیدا را rank می‌کنیم تا بهترین ۵۰ را انتخاب کنیم.

**مدل Ranking:** یک deep neural network که feature‌های بیشتری را در نظر می‌گیرد:
- Feature‌های user، item، و context را concatenate می‌کند
- Multiple dense layers
- Multi-task learning: همزمان click probability و watch-time را predict می‌کند

**Multi-task Learning:**
فقط optimize کردن برای clicks می‌تواند به clickbait منجر شود. باید چند متریک را همزمان بهینه کرد:
```
Loss = α × Click_Loss + β × WatchTime_Loss + γ × Like_Loss + δ × Share_Loss
```

**Stage 3: Business Logic و Re-ranking**

بعد از ranking، business logic اعمال می‌شود:
- **Diversity:** از یک creator یا genre بیش از ۲ ویدئو متوالی نمایش نده
- **Freshness:** ویدئوهای جدید را boost کن
- **Deduplication:** اگر کاربر ویدئو را دیده آن را حذف کن
- **Policy Filtering:** محتوای نامناسب را حذف کن

**مرحله ۴: Training Pipeline**

**Data Collection:**
- User interactions را در event stream (Kafka) log کن
- Log Processing با Spark برای تبدیل به training data
- Feature Store (Feast یا Tecton) برای serving feature‌ها هم در train و هم در inference

**Training Schedule:**
- Full retrain: هفتگی
- Incremental/online training: روزانه با داده‌های جدید
- Two-Tower model: برای retrieval می‌تواند بیشتر offline باشد
- Ranking model: نیازمند fresh data است

**مرحله ۵: Serving Architecture**

```
User Request
    ↓
Feature Service (< 5ms)
    ↓
Candidate Generation Service (< 50ms)
  ├── Collaborative Filtering
  ├── Content-based
  └── Trending/Popular
    ↓ (~3000 candidates)
Ranking Service (< 100ms)
    ↓ (top 100)
Business Logic & Re-ranking (< 20ms)
    ↓ (top 50)
Response (total < 200ms)
```

**مرحله ۶: Cold Start Problem**

برای کاربران جدید یا ویدئوهای جدید:
- **New User:** پرسیدن preferences در onboarding، استفاده از popularity-based recommendations
- **New Item:** استفاده از content features (ویدئو embedding از visual content)، exploitation مدارج اول کاربران مشابه

**مرحله ۷: Monitoring و Online Evaluation**

- **Online metrics:** CTR، Watch-time، Satisfaction rate
- **Offline metrics:** NDCG@K، MAP، Recall@K
- **A/B Testing:** هر تغییر در مدل باید آزمایش A/B داشته باشد
- **Feedback loops:** مراقب filter bubble و bias amplification باشیم

**اشتباهات رایج کاندیداها:**
- فراموش کردن two-stage pipeline و تلاش برای rank کردن ۱۰M ویدئو با یک مدل سنگین
- نادیده گرفتن cold start problem
- optimize کردن فقط برای click و نادیده گرفتن long-term engagement
- عدم اشاره به monitoring و A/B testing

**سوالات Follow-up احتمالی:**
- چطور با filter bubble و جلوگیری از over-personalization کنار می‌آیی؟
- Explore vs. Exploit چطور در recommendation system balance می‌شود؟
- چطور recommendation system را برای video series یا محتوای episodic بهینه می‌کنی؟

---

### سوال ۱۲: طراحی Real-time Fraud Detection System
**دسته:** ML System Design, Real-time Systems  
**سطح:** Senior/Staff  
**چرا مهمه:** سیستم‌های fraud detection ترکیبی از ML، engineering، و business logic هستند و نیازمند تصمیم‌گیری real-time با latency بسیار پایین هستند.  
**زمان پاسخ:** ۳۰ تا ۴۰ دقیقه

**سوال:** یک سیستم تشخیص تقلب real-time برای یک شرکت کارت اعتباری با ۱۰۰ میلیون تراکنش در روز طراحی کن. سیستم باید در کمتر از ۱۰۰ms تصمیم بگیرد و false positive rate بسیار پایین داشته باشد.

**پاسخ کامل:**

سیستم fraud detection یکی از پیچیده‌ترین ML systems است چون چندین constraint سخت دارد: latency بسیار پایین، imbalanced data، adversarial behavior (fraudsters تطبیق می‌یابند)، و هزینه بالای هر دو نوع خطا.

**مرحله ۱: Requirements Clarification**

**Scale:**
- ۱۰۰M تراکنش/روز = ~۱۱۵۰ TPS (transactions per second)
- Peak: احتمالاً ۳-۵x = ۵۰۰۰-۶۰۰۰ TPS

**Latency:**
- < ۱۰۰ms end-to-end برای reject/approve
- Real-time decision قبل از تأیید تراکنش

**Accuracy:**
- False Negative (missed fraud): کم‌ترین ممکن — مستقیم ضرر مالی
- False Positive (legitimate blocked): باید < ۰.۱٪ — تجربه کاربری بد

**مرحله ۲: Feature Engineering**

این سیستم نیازمند feature‌های real-time است که باید در لحظه محاسبه شوند:

**Transaction Features:**
- مبلغ تراکنش
- merchant category code (MCC)
- location (کشور، شهر)
- channel (online، in-person، ATM)
- time of day و day of week

**User-level Real-time Features (از Feature Store):**
- تعداد تراکنش‌های ۱، ۵، ۱۵، ۶۰ دقیقه اخیر
- مجموع مبلغ ۱، ۵، ۱۵، ۶۰ دقیقه اخیر
- تعداد merchants منحصر به فرد ۱ ساعت اخیر
- میانگین مبلغ تراکنش ۳۰ روز اخیر
- آیا این merchant قبلاً استفاده شده؟
- velocity: نرخ تراکنش نسبت به معمول

**Cross-entity Features:**
- آیا این merchant قبلاً fraud داشته؟
- merchant velocity: چند کاربر با این merchant در ۱ ساعت اخیر
- آیا این IP قبلاً مشکوک بوده؟

**مرحله ۳: معماری سیستم**

**Online Path (< 100ms):**
```
Transaction Event
      ↓
API Gateway & Validation (< 5ms)
      ↓
Feature Service — Real-time (< 15ms)
  ├── Redis (pre-computed user features)
  └── Kafka Streams (streaming aggregations)
      ↓
Rule Engine — Hard Rules (< 5ms)
  ├── Block: مبلغ > $50K برای کاربران جدید
  └── Pass: whitelist merchants
      ↓
ML Scoring Service (< 30ms)
  ├── Light Model: Gradient Boosting (LightGBM)
  └── Returns: fraud probability
      ↓
Decision Logic (< 5ms)
  ├── Score < 0.3: Approve
  ├── Score 0.3-0.7: Step-up Auth (3DS)
  └── Score > 0.7: Decline
      ↓
Response (total < 100ms)
```

**Offline Path (Training Pipeline):**
```
Transaction Events → Kafka → Spark Streaming → 
Feature Engineering → Label Creation (با تأخیر ۷-۳۰ روز) →
Model Training (daily) → Model Registry → 
A/B Testing → Production Deployment
```

**مرحله ۴: ML Models**

**Gradient Boosting (Primary Model):**
LightGBM یا XGBoost برای fraud detection بسیار مناسب است:
- سریع در inference: < ۵ms
- خوب با feature‌های mixed type
- Interpretable برای compliance و investigations

**Neural Network (Secondary Model):**
برای feature‌های sequence-based مثل تاریخچه تراکنش:
- LSTM یا Transformer روی sequence آخرین ۵۰ تراکنش
- Captures temporal patterns بهتر
- latency بیشتر (~۲۰ms)

**Ensemble:**
ترکیب هر دو مدل با weighted averaging یا stacking.

**مرحله ۵: Feature Store**

یکی از critical components این سیستم، Feature Store است:

**Online Feature Store (Redis):**
- User aggregations (velocity features) pre-computed
- Update می‌شود از Kafka Streams در real-time
- Read latency: < ۱ms

**Offline Feature Store (Spark/Hive):**
- Historical features برای training
- Point-in-time correct feature retrieval برای training data

**مرحله ۶: Handling Adversarial Behavior**

Fraudsters با مدل‌ها تطبیق پیدا می‌کنند. باید:

- **Concept Drift Detection:** Monitor کردن distribution features و fraud rate
- **Regular Retraining:** حداقل هفتگی، با real-time label feedback
- **Champion-Challenger:** نگه‌داشتن مدل قدیمی در parallel و مقایسه مداوم
- **Behavioral Biometrics:** typing pattern، device fingerprint که سخت‌تر جعل می‌شود

**مرحله ۷: Labeling Challenge**

یکی از چالش‌های اصلی این است که label‌ها با تأخیر می‌آیند:
- Chargeback: کاربر ادعای تقلب می‌کند — ممکن است ۳۰-۹۰ روز بعد اتفاق بیفتد
- Investigation: تیم fraud ops بررسی می‌کند

باید با **delayed labeling** کنار آمد:
- Early labels از rule-based triggers
- Final labels از chargeback data
- Semi-supervised learning برای unlabeled transactions

**مرحله ۸: Monitoring**

- **Business Metrics:** fraud rate، false positive rate، revenue saved
- **Model Metrics:** AUC-PR، NDCG روی recent data
- **System Metrics:** latency، error rate، feature freshness

**Explainability:**
برای compliance و investigation، باید بتوان توضیح داد چرا یک تراکنش block شد. SHAP values برای LightGBM این امکان را فراهم می‌کند.

**اشتباهات رایج کاندیداها:**
- نادیده گرفتن latency constraint و طراحی سیستم با مدل‌های سنگین
- فراموش کردن delayed labeling challenge
- عدم اشاره به feature store و real-time feature computation
- نادیده گرفتن adversarial behavior و concept drift

**سوالات Follow-up احتمالی:**
- چطور با money mule networks (graph-based fraud) کنار می‌آیی؟
- چطور مدل fraud detection را برای کشورهای مختلف personalize می‌کنی؟
- چطور cost matrix را برای decision threshold تعیین می‌کنی؟

---

### سوال ۱۳: طراحی MLOps Pipeline برای Production ML System
**دسته:** MLOps & Infrastructure  
**سطح:** Senior/Staff  
**چرا مهمه:** بیش از ۸۰٪ ML projects هرگز به production نمی‌رسند. MLOps بریج بین research و production است و شرکت‌های بزرگ به آن به‌شدت اهمیت می‌دهند.  
**زمان پاسخ:** ۲۵ تا ۳۵ دقیقه

**سوال:** یک MLOps pipeline کامل برای یک ML model که باید در production deploy شود و به‌صورت مداوم بهبود یابد طراحی کن. از data ingestion تا monitoring در production را پوشش بده.

**پاسخ کامل:**

MLOps (Machine Learning Operations) اعمال اصول DevOps، DataOps، و ModelOps به سیستم‌های یادگیری ماشین است. هدف اصلی، ایجاد یک cycle مداوم از توسعه، deployment، monitoring، و بهبود مدل است.

**معماری کلی MLOps Pipeline:**

```
Data Layer → Feature Layer → Training Layer → Evaluation Layer → Serving Layer → Monitoring Layer
    ↑_______________________________________Feedback Loop____________________________________|
```

**بخش ۱: Data Pipeline**

**Data Ingestion:**
- Batch ingestion: Spark/Airflow برای historical data
- Stream ingestion: Kafka/Kinesis برای real-time data
- Data validation با Great Expectations:
  - Schema validation
  - Range checks
  - Distribution checks (data drift detection)

**Data Versioning:**
- **DVC (Data Version Control):** مثل Git برای data
- **Delta Lake / Iceberg:** ACID transactions و time travel برای data lake
- هر dataset باید versioned و reproducible باشد

**Data Catalog:**
- هر dataset باید documented باشد: منبع، schema، freshness، owner
- ابزارها: Apache Atlas، DataHub، Amundsen

**بخش ۲: Feature Engineering و Feature Store**

Feature Store یکی از مهم‌ترین اجزای MLOps است:

**چرا Feature Store لازم است؟**
- جلوگیری از training-serving skew: feature‌های training و inference باید یکسان باشند
- Feature reuse: تیم‌های مختلف می‌توانند از feature‌های یکدیگر استفاده کنند
- Point-in-time correct retrieval: برای training باید فقط به داده‌های قبل از prediction time دسترسی داشت

**Components:**
- **Offline Store** (Hive/BigQuery): برای training
- **Online Store** (Redis): برای low-latency inference
- **Sync Pipeline**: داده را از offline به online sync می‌کند

**بخش ۳: Model Training**

**Experiment Tracking:**
هر experiment باید کاملاً track شود:
- کد (git commit hash)
- hyperparameters
- dataset version
- metrics
- artifacts (model files)

ابزار: MLflow، Weights & Biases، Neptune

```python
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001, "epochs": 100})
    mlflow.log_metrics({"val_auc": 0.95})
    mlflow.sklearn.log_model(model, "model")
```

**Hyperparameter Optimization:**
- Optuna یا Ray Tune برای automated HPO
- Population-based training برای neural networks

**Distributed Training:**
- Horovod یا PyTorch DDP برای data parallel training
- DeepSpeed برای very large models

**بخش ۴: Model Evaluation و Validation**

قبل از deploy کردن، چندین gate باید pass شود:

**Automated Testing:**
- **Unit tests:** هر function در feature engineering
- **Integration tests:** pipeline از input تا output
- **Model tests:**
  - Performance روی held-out test set بالاتر از threshold است؟
  - Performance روی slices (demographic groups) قابل قبول است؟
  - Model هیچ bias مضری ندارد؟

**Shadow Mode:**
مدل جدید در parallel با مدل قدیمی اجرا می‌شود اما prediction‌هایش به production نمی‌رسد. فقط log می‌شود تا مقایسه شود.

**Canary Deployment:**
ابتدا ۱٪ traffic به مدل جدید می‌رود. اگر metrics خوب بود، به‌تدریج افزایش می‌یابد.

**بخش ۵: Model Registry**

مدل Registry جایی است که مدل‌های trained نگهداری می‌شوند:
- هر مدل versioned است
- State‌ها: Development → Staging → Production → Archived
- هر مدل با metadata کامل: کد، داده، metrics

ابزار: MLflow Model Registry، SageMaker Model Registry

**بخش ۶: Serving و Deployment**

**Deployment Strategies:**
- **Blue-Green:** دو محیط identical، traffic switch یک‌باره
- **Canary:** تدریجی — ریسک کمتر
- **Rolling:** سرورها یک به یک update می‌شوند
- **Feature Flags:** می‌توان مدل را در code بدون deploy تغییر داد

**Serving Patterns:**
- **Online Serving:** REST API با FastAPI/TorchServe
- **Batch Scoring:** Spark job برای offline predictions
- **Streaming:** Kafka + Flink برای real-time scoring

**بخش ۷: Monitoring**

این بخش اغلب نادیده گرفته می‌شود اما حیاتی است:

**Technical Metrics:**
- Latency (p50، p90، p99)
- Throughput و Error Rate
- Model resource usage (CPU، GPU، Memory)

**Data Drift Detection:**
- **Feature drift:** distribution input features تغییر کرده؟
- روش: Population Stability Index (PSI) یا Kolmogorov-Smirnov test
- Alert وقتی PSI > 0.2

**Model Performance Monitoring:**
- اگر labels با تأخیر می‌آیند: proxy metrics (مثل engagement به‌جای conversion)
- اگر labels real-time هستند: monitor accuracy/AUC directly

**Concept Drift:**
- رابطه بین features و target عوض شده
- سخت‌ترین نوع drift برای detection

**بخش ۸: Retraining Triggers**

- **Scheduled:** هر هفته/ماه — ساده اما ممکن است drift را دیر بگیرد
- **Performance-based:** وقتی metric از threshold پایین‌تر رفت
- **Data-based:** وقتی data drift تشخیص داده شد
- **Volume-based:** وقتی به اندازه کافی داده جدید جمع شد

**بخش ۹: Governance و Compliance**

در صنایع regulated (financial، healthcare):
- Model documentation و explainability
- Audit trail از هر prediction
- Fairness metrics و bias testing
- Model approval process

**ابزارهای محبوب در هر بخش:**
- Orchestration: Apache Airflow، Prefect، Kubeflow Pipelines
- Experiment Tracking: MLflow، W&B
- Feature Store: Feast، Tecton، Vertex AI Feature Store
- Model Registry: MLflow، SageMaker
- Serving: TorchServe، Triton, KServe، Ray Serve
- Monitoring: Evidently AI، WhyLabs، Arize AI
- Infrastructure: Kubernetes، Terraform

**اشتباهات رایج کاندیداها:**
- تمرکز فقط بر training و فراموش کردن monitoring
- نادیده گرفتن training-serving skew
- عدم اشاره به model registry و versioning
- فراموش کردن governance و compliance در regulated industries

**سوالات Follow-up احتمالی:**
- چطور با هزار مدل مختلف در production کنار می‌آیی؟
- Online learning vs. batch retraining — trade-off چیست؟
- چطور ROI یک MLOps investment را اندازه می‌گیری؟

---

### سوال ۱۴: طراحی Large-Scale Image Classification System
**دسته:** Computer Vision System Design  
**سطح:** Senior  
**چرا مهمه:** CV systems در شرکت‌هایی مثل Google Photos، Instagram، Pinterest، و Amazon در مقیاس بسیار بزرگ عمل می‌کنند. طراحی این سیستم‌ها نیازمند درک عمیق از DL architecture و distributed systems است.  
**زمان پاسخ:** ۲۵ تا ۳۵ دقیقه

**سوال:** یک سیستم Content Moderation طراحی کن که بطور خودکار تصاویر نامناسب را در یک social media platform با ۵۰۰ میلیون کاربر و ۱۰۰ میلیون تصویر آپلود شده در روز شناسایی کند.

**پاسخ کامل:**

Content moderation یکی از پیچیده‌ترین چالش‌های AI در شرکت‌های بزرگ است چون باید accuracy بالا، latency پایین، و در عین حال fairness را حفظ کند — در حالی که adversaries مدام سعی می‌کنند از سیستم فرار کنند.

**مرحله ۱: Requirements**

**Scale:**
- ۱۰۰M image/day ≈ ۱۱۵۰ image/sec (average), peak ~۵x = ۵۷۵۰/sec
- هر تصویر باید در < ۵ ثانیه بررسی شود (قبل از نمایش به عموم)

**Categories:**
- NSFW/Adult content
- Violence و Graphic content
- Hate speech و Harassment symbols
- Spam و Scam
- Child Safety (CSAM) — highest priority

**متریک‌های Success:**
- Recall بالا برای CSAM: > ۹۹.۹٪ (هیچ‌چیز miss نشود)
- Precision قابل قبول: < ۲٪ false positive rate (کاربران معمولی block نشوند)

**مرحله ۲: Data Pipeline**

**Training Data:**
- ImageNet برای pretraining
- Platform-specific labeled data
- CSAM: PhashDB (perceptual hash database)، PhotoDNA
- Human annotation با strict safety protocols برای annotators

**Data Augmentation:**
- Random flips، crops، color jitter
- Mixup، CutMix برای robustness
- Adversarial augmentation برای مقاومت در برابر evasion

**مرحله ۳: Model Architecture**

**Stage 1: Fast Pre-filter (< 10ms)**

یک مدل بسیار سبک برای سریع رد کردن تصاویر clearly safe:
- MobileNet-V3 یا EfficientNet-Lite quantized به INT8
- Binary classification: safe vs. needs_review
- ~۹۵٪ تصاویر در این مرحله pass می‌شوند

**Stage 2: Multi-label Classification (< 50ms)**

برای تصاویر که از stage 1 رد نشدند:
- EfficientNet-B4 یا ViT-B fine-tuned
- Multi-label: هر تصویر می‌تواند چند label داشته باشد
- Output: probability برای هر category

**Stage 3: Specialized Models (< 100ms)**

برای categories با sensitivity بالا:
- **CSAM Detection:** PhotoDNA-based hash matching + dedicated classifier
- **Face Detection + Age Estimation:** برای تشخیص ماینور
- **OCR + Text Analysis:** برای تشخیص hate speech در متن داخل تصویر

**مرحله ۴: Perceptual Hashing**

برای تصاویر already-known (مثل CSAM که قبلاً report شده):
- **pHash:** resistant به minor modifications
- **PhotoDNA:** Microsoft's technology برای CSAM matching
- hash database را lookup کنیم — < ۱ms

**مرحله ۵: Serving Architecture**

```
Upload Request
    ↓
CDN → Storage (S3/GCS)
    ↓
Content Moderation Queue (Kafka)
    ↓
Orchestrator Service
    ├── Hash Matching Service (< 1ms) — block if match
    ├── Fast Pre-filter (< 10ms) — if safe, approve
    ├── Multi-label Classifier (< 50ms) — score all categories
    └── Specialized Classifiers (async, < 100ms)
          ↓
Decision Engine
    ├── Auto-approve: all scores < low_threshold
    ├── Auto-reject: any score > high_threshold  
    └── Human Review Queue: scores in middle range
          ↓
Action: Approve / Remove / Shadow-ban / Escalate
```

**مرحله ۶: Human Review Queue**

ML model نمی‌تواند همه چیز را با confidence بالا تصمیم بگیرد:
- تصاویر با scores در middle range به human review می‌روند
- **Active Learning:** reviewers برگزیده‌ترین samples برای labeling هستند
- **Reviewer Wellbeing:** rotation، psychological support
- **Quality Control:** spot-checking و inter-annotator agreement

**مرحله ۷: Adversarial Robustness**

Adversaries از روش‌های مختلف سعی می‌کنند مدل را bypass کنند:
- Image flipping، rotation، small perturbations
- Adding noise یا watermarks
- Steganography

**راه‌حل‌ها:**
- Adversarial training: اضافه کردن adversarial examples در training
- Ensemble models: استفاده از چند مدل مختلف
- Behavioral signals: اگر یک account مدام تصاویر مشکوک upload می‌کند، scrutiny بیشتر
- Hash normalization: قبل از hash matching، image را normalize کن

**مرحله ۸: Fairness و Bias**

یکی از چالش‌های مهم، bias در مدل است:
- مدل‌ها ممکن است برای گروه‌های demographic خاص false positive بیشتری داشته باشند
- باید performance را روی slices مختلف جداگانه monitor کرد
- Regular fairness audits

**مرحله ۹: Appeals Process**

کاربران باید بتوانند تصمیم‌ها را appeal کنند:
- Human review team برای appeals
- این feedback می‌تواند به بهبود مدل کمک کند

**مرحله ۱۰: Compliance**

- **CSAM:** باید به NCMEC (National Center for Missing & Exploited Children) report شود
- **GDPR:** data retention policies
- **Audit logs:** هر تصمیم باید traceable باشد

**اشتباهات رایج کاندیداها:**
- فراموش کردن multi-stage pipeline — تلاش برای همه چیز در یک مدل
- نادیده گرفتن hash-based matching برای known content
- فراموش کردن human review و appeals process
- عدم توجه به fairness و bias

**سوالات Follow-up احتمالی:**
- چطور با محتوای video در مقابل image کنار می‌آیی؟
- چطور با evolving content policies و categories جدید کنار می‌آیی؟
- چطور latency و recall را برای CSAM vs. NSFW trade-off می‌کنی؟

---

### سوال ۱۵: طراحی Search Ranking System
**دسته:** Search & Information Retrieval  
**سطح:** Staff  
**چرا مهمه:** Search ranking در قلب Google، Bing، Amazon، LinkedIn، و تقریباً هر شرکت بزرگ است. این یکی از پیچیده‌ترین و جالب‌ترین ML system design سوالات است.  
**زمان پاسخ:** ۳۵ تا ۴۵ دقیقه

**سوال:** یک سیستم search ranking برای یک e-commerce platform مثل Amazon طراحی کن. کاربران query می‌زنند و باید مرتبط‌ترین و بهترین محصولات نمایش داده شوند. معماری ML pipeline، feature engineering، و online learning را توضیح بده.

**پاسخ کامل:**

Search ranking یکی از پیچیده‌ترین ML problems است چون باید چندین objective را همزمان optimize کند: relevance، user engagement، revenue، و user satisfaction — و این objective‌ها گاهی با هم conflict دارند.

**مرحله ۱: Requirements**

**Goals:**
- کاربر محصول مورد نظرش را سریع پیدا کند
- Conversion rate بالا (کلیک → خرید)
- Revenue برای platform (با در نظر گرفتن sponsored results)

**Scale:**
- ۱ میلیون query/روز، peak ۵۰۰ QPS
- ۵۰۰ میلیون محصول
- Latency: < ۱۰۰ms

**مرحله ۲: معماری کلی**

```
Query
  ↓
Query Understanding
  ├── Spell correction
  ├── Query expansion (synonyms)
  └── Intent classification
  ↓
Retrieval (از 500M به ~1000)
  ├── Inverted Index (BM25)
  ├── Dense Retrieval (Two-Tower)
  └── Business Rules (boost new products)
  ↓
Ranking (~1000 به top 20)
  ├── Relevance Score
  ├── Predicted CTR
  └── Predicted Conversion Rate
  ↓
Business Logic
  ├── Sponsored Results Insertion
  ├── Diversity
  └── Policy Filters
  ↓
Results
```

**مرحله ۳: Query Understanding**

قبل از هر چیز باید query را بفهمیم:

**Spell Correction:**
- Dictionary-based + ML-based
- Context-aware: «apple watch» با «aple watch» فرق دارد با «aple» در یک query دیگر

**Query Expansion:**
- Synonym expansion با Word2Vec/FastText trained روی query logs
- Category expansion: «laptop» → «notebook computer» + «ultrabook»

**Query Classification:**
- **Navigational:** کاربر یک برند خاص می‌خواهد (Nike shoes)
- **Informational:** کاربر اطلاعات می‌خواهد
- **Transactional:** کاربر می‌خواهد خرید کند

**مرحله ۴: Feature Engineering**

**Query Features:**
- Query embedding (BERT-based)
- Query length، uniqueness
- Query category
- Historical CTR برای این query

**Product Features:**
- Title، description، brand embedding
- Category (hierarchical)
- Price، rating، review count
- Inventory status
- Historical engagement (CTR، conversion rate)
- Freshness (چه موقع آمده)

**Query-Product Match Features (Cross Features):**
- BM25 score: keyword overlap
- Semantic similarity: embedding similarity
- Exact match signals: query words در title چند بار؟
- Query-category match: آیا محصول در expected category است؟

**User × Query × Product Features (Personalization):**
- آیا کاربر قبلاً این محصول را دیده؟
- آیا کاربر از این brand خریده؟
- User price sensitivity
- User category preferences

**Context Features:**
- Device type (mobile vs. desktop)
- Time of day
- Location (برای shipping time estimation)
- Session context (چه چیزهایی قبلاً جستجو کرده)

**مرحله ۵: ML Models**

**Retrieval Stage:**

**BM25 (Sparse Retrieval):**
```
BM25(q,d) = Σ IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))
```

**Dense Retrieval (Two-Tower):**
- Query encoder: BERT → 256-dim embedding
- Product encoder: Title + features → 256-dim embedding
- در inference: query embedding را محاسبه کن، سپس با FAISS nearest neighbor search

**Ranking Stage:**

**Learning to Rank:**

سه رویکرد اصلی:
- **Pointwise:** مثل regression، هر query-product pair را مستقل score می‌دهد
- **Pairwise:** مثل RankNet، تلاش می‌کند جفت‌های (doc_i, doc_j) را درست ترتیب دهد
- **Listwise:** مثل LambdaMART، کل list را optimize می‌کند

در عمل، **LambdaMART** (که پایه LightGBM ranking است) یکی از قوی‌ترین مدل‌ها است.

**Multi-task Learning:**

همزمان predict می‌کنیم:
- CTR (Click-through Rate): آیا کاربر کلیک می‌کند؟
- CVR (Conversion Rate): آیا خرید انجام می‌شود؟
- Revenue: چقدر revenue generate می‌شود؟
- Satisfaction: آیا کاربر راضی است؟

```
Final Score = α × predicted_CTR + β × predicted_CVR + γ × revenue_value
```

**Position Bias:**

کاربران بیشتر روی نتایج بالاتر کلیک می‌کنند صرفاً به‌خاطر position. باید این را در training correction کرد:
- **Inverse Propensity Weighting:** هر click را با inverse position probability وزن می‌دهیم
- **Unbiased Learning to Rank**

**مرحله ۶: Training Data**

داده‌های training از user interactions جمع می‌شود:
- **Positive:** کلیک → خرید (قوی‌ترین signal)
- **Weak Positive:** کلیک بدون خرید
- **Negative:** نمایش بدون کلیک
- **Strong Negative:** skip کردن نتایج بالاتر

**Implicit Feedback:**
- Dwell time روی صفحه محصول
- Add to cart بدون خرید
- Scroll depth در صفحه نتایج

**مرحله ۷: Online Learning و Freshness**

Search ranking باید به تغییرات سریع واکنش نشان دهد:
- محصول جدید (هنوز click data ندارد — exploration)
- Trend (یک کلمه trending می‌شود)
- Inventory change (محصول تمام شد)

**Online Learning:**
- مدل‌های gradient boosting می‌توانند به‌صورت incremental update شوند
- Neural ranking models نیازمند batch retraining هستند (daily)

**مرحله ۸: Evaluation**

**Offline Metrics:**
- **NDCG@k (Normalized Discounted Cumulative Gain):** متریک استاندارد برای ranking
- **MAP (Mean Average Precision)**
- **MRR (Mean Reciprocal Rank)**

**Online Metrics:**
- CTR، CVR، Revenue per query
- Customer satisfaction (surveys)
- Zero-result rate (query که نتیجه ندارد)

**مرحله ۹: Sponsored Results و Revenue Optimization**

یکی از چالش‌های واقعی، ترکیب organic results با sponsored است:
- Sponsored results باید مرتبط هم باشند — Google Ad Quality Score
- **vCPM (viewable cost per million)** و **eCPM** برای قیمت‌گذاری
- باید مطمئن شویم sponsored results تجربه کاربر را خراب نمی‌کند

**اشتباهات رایج کاندیداها:**
- نادیده گرفتن position bias در training data
- فراموش کردن multi-task learning (فقط CTR optimize کردن)
- عدم اشاره به Learning to Rank vs. simple regression
- نادیده گرفتن cold-start برای محصولات جدید

**سوالات Follow-up احتمالی:**
- چطور search را برای voice queries بهینه می‌کنی؟
- چطور با long-tail queries (خیلی نادر) کنار می‌آیی؟
- چطور privacy-preserving ranking طراحی می‌کنی؟

---

*پایان سوالات روز ۲۰ آوریل ۲۰۲۶ — ۱۵ سوال جامع در ۳ حوزه Data Science، AI Engineering، و DL/ML System Design*

