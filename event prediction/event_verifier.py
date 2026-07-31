import time
import json
import random
import re
import gc
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class QwenEventVerifier:
    """
    Uses a 4-bit-quantized Qwen LLM to check whether a candidate cluster
    actually represents one well-defined event: samples up to `sample_size`
    of its messages and asks the model to answer in JSON (is_event/title/
    summary). Only the Python-side logging/safety code has been touched
    here - the system prompt sent to the model is left exactly as given,
    since it's the tuned instruction the model is judged against.
    """

    def __init__(self, model_path="../../models/qwen/", sample_size=10,
                 max_new_tokens=256, max_input_tokens=4800,
                 max_consecutive_oom_before_abort=5):
        self.model_path = model_path
        self.sample_size = sample_size
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.max_consecutive_oom_before_abort = max_consecutive_oom_before_abort
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("\n" + "=" * 60)
        print(f"[QwenVerifier] STARTING: loading Qwen model (4-bit) at {_now()}")
        print("=" * 60)
        print(f"[QwenVerifier] Model path: {self.model_path}")
        print(f"[QwenVerifier] torch.cuda.is_available(): {torch.cuda.is_available()}")

        compute_dtype, reason = self._select_compute_dtype(self.device)
        print(f"[QwenVerifier] 4-bit compute dtype: {compute_dtype} - {reason}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )

        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map={"": 0} if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"[QwenVerifier] ERROR loading model from '{self.model_path}': {e}")
            print("[QwenVerifier] If this is an out-of-memory error or a quantization "
                  "support error, the model may be too large (in 4-bit) for the free "
                  "VRAM on this GPU, or this GPU/driver/bitsandbytes combination may not "
                  "support 4-bit inference for this architecture.")
            raise
        t1 = time.time()
        print(f"[QwenVerifier] Model + tokenizer loaded in {t1 - t0:.2f}s")

        if self.device.type == "cuda":
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[QwenVerifier] VRAM used by the model: {vram:.2f} GB")

        print("[QwenVerifier] READY.")
        print("=" * 60 + "\n")

    @staticmethod
    def _select_compute_dtype(device: torch.device):
        if device.type != "cuda":
            return torch.float32, "no GPU available, using float32 on CPU"
        major, minor = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)
        if major >= 7:
            return torch.float16, f"{device_name} (compute {major}.{minor}) has fast fp16"
        if major == 6 and minor == 0:
            return torch.float16, f"{device_name} (compute {major}.{minor}) is P100-class, fast fp16"
        return torch.float32, (f"{device_name} (compute {major}.{minor}) has crippled fp16 "
                                f"throughput on consumer Pascal/older GPUs - using float32 "
                                f"as the 4-bit compute dtype")
    @staticmethod
    def _clean_message(text):
        """Removes links, mentions, and emojis from the text."""
        # حذف لینک‌ها (شامل http، www، تلگرام و توییتر/X)
        text = re.sub(r'(https?://\S+|www\.\S+|t\.me/\S+|twitter\.com/\S+|x\.com/\S+)', ' ', text)
        # حذف منشن‌ها
        text = re.sub(r'@\w+', ' ', text)
        # حذف ایموجی‌ها (با استفاده از بازه‌های یونیکد متداول ایموجی‌ها)
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # حذف فاصله‌های اضافی ایجاد شده
        return re.sub(r'\s+', ' ', text).strip()

    def _build_prompt(self, messages):
        system_prompt = """
        تو یک ارزیاب متخصص در شناسایی «خبرهای برنامه‌ریزی‌شده‌ی مهمِ آینده» از متن فارسی هستی.
به تو یک یا چند پیام (که در یک خوشه‌ی موضوعی مشابه قرار گرفته‌اند) داده می‌شود.

### تعریف رویداد
خوشه «خبر مهم آینده» است اگر همزمان این سه شرط برقرار باشد:
1. به یک اتفاق مشخص و برنامه‌ریزی‌شده اشاره دارد (نه گزارش کلی/تحلیل).
2. وقوع آن در **آینده** است — حداکثر تا یک ماه از تاریخ پیام. اگر رویداد قبلا برگزار شده یا گذشته است، رویداد محسوب نمی‌شود.
3. یک **کنشگر مهم** در آن درگیر است: نهاد/مقام رسمی دولتی، وزارتخانه، تیم یا باشگاه ورزشی شناخته‌شده، سازمان یا شرکت بزرگ، یا شخصیت سیاسی/مذهبی/فرهنگی شناخته‌شده. (فرد عادی، کسب‌وکار محلی کوچک، یا آگهی تبلیغاتی کنشگر مهم محسوب نمی‌شود.)

نمونه‌های رویداد: کنسرت بزرگ (مثل بلوار آزادی)، دیدار مهم فوتبال قبل از برگزاری، دیدار یا نشست دولتی/دیپلماتیک، جلسه رسمی مهم، افتتاحیه‌ی بزرگ، انتخابات، مهلت یا تاریخ اجرای یک تصمیم رسمی.

### موارد خارج از تعریف (باید رد شوند)
1. مهم: پیش بینی های هواشناسی مربوط به اینده رویداد نیست.
2. مهم: جدول قطعی برق مربوط به اینده رویداد نیست
1 آگهی استخدام یا تبلیغات تجاری/شرط‌بندی
2. گزارش‌های روتین و تکرارشونده (پیش‌بینی هواشناسی، برنامه‌ی خاموشی برق، برنامه‌ی هفتگی ثابت)
3. تحلیل، اظهارنظر یا واکنش سیاسی/اجتماعی که خودش گزارشِ رخداد جدیدی نیست
4. پیام تبریک، دلنوشته، یا محتوای انگیزشی عمومی
5. محتوای تاریخی/مناسبتی صرف (نقل‌قول قدیمی، تقویم تاریخ، زندگی‌نامه)
6. گزارش مالی یا آماری دوره‌ای
7. خبری که به اتفاقی که قبلاً برگزار شده یا در گذشته اتفاق افتاده
8. اتفاقی که کنشگر مهم در آن درگیر نیست (مثلاً یک مراسم محلی کوچک بدون نهاد/شخصیت شناخته‌شده)
در صورت کوچک‌ترین تردید، false برگردان. باز هم تاکید میشود که پیش بینی هواشناسی و جدول قطعی برق و موارد این شکل رویداد نیست.
پاسخ فقط باید یک JSON معتبر باشد، بدون هیچ متن اضافه.
در صورت رد:
{
  "is_event": false,
  "title": null,
  "summary": null
}

در صورت تأیید:
{
  "is_event": true,
  "title": "عنوان کوتاه و دقیق رویداد",
  "summary": "خلاصه یک یا دو جمله‌ای از رویداد، شامل کنشگر مهم و تاریخ/بازه‌ی زمانی"
}"""
        
#         system_prompt = """
# تو یک ارزیاب متخصص در شناسایی «رویداد خبری» از متن فارسی هستی.
# به تو یک یا چند پیام (که در یک خوشه‌ی موضوعی مشابه قرار گرفته‌اند) داده می‌شود.
# باید مشخص کنی که آیا این خوشه یک «رویداد» که در آینده اتفاق خواهد افتاد است یا نه.
# ### تعریف رویداد
# خوشه «رویداد» است اگر پیام‌هایش به یک اتفاق مشخص، زمان‌مند و غیرتکرارشونده در دنیای واقعی اشاره کنند —حتما باید به وقوع آن رویداد در آینده‌ی نزدیک اشاره شده باشد. اگر خبر از برگزاری و گذشته بود رویداد نیست. ماژول پیش بینی رویداد است پس باید اتفاق یا رخداد به آینده اشاره کند. اما قطعی برق و پیش بینی اب و هوا در این چارچوب نمی‌گنجد.
# ### موارد خارج از تعریف رویداد (باید رد شوند)
# 1. آگهی استخدام یا تبلیغات تجاری/شرط‌بندی
# 2. گزارش‌های روتین و تکرارشونده بدون ویژگی خاص (پیش‌بینی هواشناسی روزانه، برنامه‌ی معمول خاموشی برق، برنامه‌ی هفتگی ثابت یک برنامه مذهبی/فرهنگی)
# 3. تحلیل، اظهارنظر یا واکنش سیاسی/اجتماعی که خودش گزارشِ رخداد جدیدی نیست
# 4. پیام تبریک، دلنوشته، یا محتوای انگیزشی عمومی
# 5. محتوای تاریخی/مناسبتی صرف (نقل‌قول‌های قدیمی، تقویم تاریخ، زندگی‌نامه) که گزارشی از مراسم/رخداد جاری نیست
# 6. گزارش‌های مالی یا آماری دوره‌ای (صورت‌وضعیت صندوق، گزارش بورس روزانه)
# 7. اشاره به رویداد برگزار شده یا در گذشته اتفاق افتاده

# در صورت کوچک‌ترین تردید، false برگردان.

# پاسخ فقط باید یک JSON معتبر باشد.

# در صورت رد:
# {
#   "is_event": false,
#   "title": null,
#   "summary": null
# }

# در صورت تأیید:
# {
#   "is_event": true,
#   "title": "عنوان کوتاه و دقیق رویداد",
#   "summary": "خلاصه یک یا دو جمله‌ای از رویداد"
# }
# """
        def render(msgs):
            numbered_messages = "\n".join(
                [f"[{i}] {msg.strip()}" for i, msg in enumerate(msgs, 1)]
            )

            user_content = f"""لطفاً پیام‌های زیر را که داخل تگ <messages> قرار دارند با دقت بررسی کن:

<messages>
{numbered_messages}
</messages>

آیا این پیام‌ها یک رویداد واحد، مهم و کلان را نشان می‌دهند؟
فقط و فقط یک JSON تولید کن. هیچ کلمه‌ای قبل یا بعد از آکولادها ننویس."""

            messages_chat = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()}
            ]

            candidate_prompt = self.tokenizer.apply_chat_template(
                messages_chat,
                tokenize=False,
                add_generation_prompt=True
            )
            # ترفند تزریق دستی آکولاد باز برای مجبور کردن مدل به تولید JSON ساختاریافته
            candidate_prompt += "\n{"

            token_count = len(self.tokenizer(candidate_prompt, add_special_tokens=False).input_ids)
            return candidate_prompt, token_count

        # تلاش اول: همه‌ی پیام‌ها با طول کامل.
        candidate_prompt, token_count = render(messages)
        if token_count <= self.max_input_tokens:
            return candidate_prompt, list(messages)

        # جا نشد: به‌جای حذف کامل پیام‌ها از انتها (رفتار قبلی)، هر پیام را
        # به‌تدریج کوتاه‌تر می‌کنیم - دقیقاً همان فلسفه‌ی مسیر ریکاوری OOM در
        # analyze_cluster_with_retry. حذف کامل پیام‌ها باعث می‌شد خوشه‌هایی که
        # برای جا شدن در بودجه‌ی توکن به ۲-۳ پیام تقلیل پیدا می‌کردند، تقریباً
        # همیشه به چشم مدل «واضحاً یک رویداد یکتا» بیایند - چون چیزی از سایر
        # پیام‌های خوشه باقی نمی‌ماند که با آن مخالفت کند یا نشان دهد خوشه
        # ناهمگن/پراکنده است. کوتاه کردن هر پیام، برخلاف حذفش، تصویر کلی خوشه
        # (تعداد پیام، تنوع منابع، ناهمگنی احتمالی) را دست‌نخورده نگه می‌دارد.
        for cap in (400, 300, 200, 150):
            msgs = [self._truncate_message_tokens(m, cap) for m in messages]
            candidate_prompt, token_count = render(msgs)
            if token_count <= self.max_input_tokens:
                print(f"[QwenVerifier]   Capped each of the {len(messages)} message(s) at "
                      f"{cap} token(s) (none dropped) so the prompt - including the forced "
                      f"'{{' - stays under max_input_tokens={self.max_input_tokens}.")
                return candidate_prompt, msgs

        # آخرین راه‌حل (تقریباً هیچ‌وقت نباید به اینجا برسد): حتی با کوتاه‌ترین
        # سقف هم جا نشد (مثلاً sample_size خیلی بزرگ کنار max_input_tokens
        # خیلی کوچک) - فقط حالا، و فقط به‌عنوان آخرین چاره، از انتها پیام حذف
        # می‌کنیم، آن هم روی نسخه‌ی از قبل کوتاه‌شده به ۳۰ توکن.
        msgs = [self._truncate_message_tokens(m, 200) for m in messages]
        while len(msgs) > 1:
            msgs = msgs[:-1]
            candidate_prompt, token_count = render(msgs)
            if token_count <= self.max_input_tokens:
                print(f"[QwenVerifier]   WARNING: even 200-token-capped messages didn't fit "
                      f"under max_input_tokens={self.max_input_tokens} - dropped down to "
                      f"{len(msgs)}/{len(messages)} message(s) as a last resort.")
                return candidate_prompt, msgs

        print(f"[QwenVerifier]   WARNING: even a single 200-token-capped message doesn't fit "
              f"under max_input_tokens={self.max_input_tokens} ({token_count} tokens) - "
              f"sending it as-is; it may still get cut off.")
        return candidate_prompt, msgs
    
    def analyze_cluster(self, messages):
        prompt, _used_messages = self._build_prompt(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        ).to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except RuntimeError:
            # If generate() OOMs, `inputs` is still sitting on the GPU and
            # nothing below this point runs to free it. Deleting it here,
            # in this same frame, drops the reference immediately; if we
            # left this to the caller, the exception's traceback keeps this
            # frame (and `inputs`) alive until the exception object itself
            # is discarded, so even a torch.cuda.empty_cache() right after
            # catching it wouldn't actually reclaim this memory. That's
            # what was making the shrink-and-retry loop still fail on the
            # very next attempt - each failed attempt was leaking its input
            # tensor instead of freeing it.
            #
            # This print is the ONLY place an OOM'd prompt's token count
            # gets logged - the normal "VRAM after generate()" line below
            # only runs on SUCCESS, so up to now a failing attempt gave no
            # visibility into how big the prompt actually was when it blew
            # up. Needed to tell apart "prompt genuinely too big for this
            # GPU" from "something else is eating the VRAM".
            prompt_tokens = inputs.input_ids.shape[1]
            if torch.cuda.is_available():
                vram_now = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"[QwenVerifier]   OOM'd with a {prompt_tokens}-token prompt | "
                      f"VRAM allocated: {vram_now:.2f} GB, reserved: {vram_reserved:.2f} GB "
                      f"right before the crash")
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        if self.device.type == "cuda":
            vram_now = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[QwenVerifier]   VRAM after generate(): {vram_now:.2f} GB "
                  f"(peak so far: {vram_peak:.2f} GB) | prompt was "
                  f"{inputs.input_ids.shape[1]} token(s)")

        # بازگرداندن آکولاد باز شده به ابتدای رشته برای جلوگیری از خطای پارس JSON
        if not response.startswith("{"):
            response = "{" + response

        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    def _truncate_message_tokens(self, text, max_tokens):
        """Cuts a single message down to its first `max_tokens` tokens
        (using the model's own tokenizer, not a word-count approximation) -
        the message stays in the sample, just shorter. Token count is what
        actually drives VRAM usage, so this is a more precise knob than
        cutting by word count."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def analyze_cluster_with_retry(self, messages, cluster_label="cluster", max_retries=5,
                                    oom_token_caps=(300, 250, 200, 150)):
        """
        Wraps analyze_cluster() with the retry + OOM-recovery + JSON-parse
        loop: tries up to `max_retries` times FOR THIS ONE CLUSTER. This is
        the ONLY place this logic should live - verify_candidates() and any
        other caller (e.g. qwen_tester.py's analyze_all_candidates()) call
        this instead of analyze_cluster() directly, so a bare try/except
        that gives up after one OOM never happens anywhere.

        On OOM, every message in the cluster is TRUNCATED to its first N
        tokens instead of dropping messages from the cluster outright.
        Dropping whole messages was an earlier approach, but it backfired:
        a cluster with only 1-2 messages left almost always reads as
        "clearly one event" to the model, since there's nothing left to
        disagree with it - so OOM'd clusters were getting silently
        rubber-stamped as confirmed events far more often than clusters
        that never OOM'd in the first place. Truncating keeps every
        message represented, just shorter.

        EXCEPTION to the above: if a cap change produces the exact same
        (already-short) messages as the previous attempt, capping harder
        is provably useless - the bottleneck isn't message length (most
        likely it's the fixed system-prompt overhead), so retrying would
        just reproduce the identical OOM. Only THEN does this fall back to
        dropping the single longest remaining message, one at a time, since
        that's the only thing left that actually shrinks the prompt.

        `oom_token_caps` is walked through in order on each successive OOM
        for the SAME cluster: attempt 1 sends messages at full length; the
        1st OOM caps each message at oom_token_caps[0] (300) tokens; the
        2nd OOM at oom_token_caps[1] (200) tokens. If it OOMs again after
        the last configured step, the cap keeps halving from there (100,
        50, ...) as a fallback so it doesn't just retry at 200 forever -
        say if you'd rather it stay at 200 instead, that's an easy change.

        IMPORTANT - this function does NOT decide whether to abort the
        whole run. It used to take a `consecutive_oom_count` in/out and
        return `should_abort` itself, sharing the exact same counter (and
        the exact same value, 5) as `max_retries` - which meant a single
        stubborn cluster that failed all of ITS OWN retries would, on its
        own, immediately trip the "abort the entire run" threshold, since
        both were counting the same 5 attempts. That's why one hard
        cluster could kill a 278-candidate run before candidate #2 ever
        ran. Deciding to abort the run is now entirely the CALLER's job
        (verify_candidates / analyze_all_candidates): it should keep its
        OWN counter of how many CANDIDATES in a row came back with
        parsed_json=None, and only abort after several DIFFERENT clusters
        in a row fail completely - not after one cluster uses up its own
        retry budget.

        Returns a dict:
            {
                'parsed_json': dict or None (None = every attempt failed),
                'raw_response': str or None (last raw output, if any),
                'used_token_cap': int or None (None = sent at full length),
                'attempts': int (how many attempts were actually made),
                'oom_count': int (how many of those attempts were OOMs),
            }
        """
        messages = [self._clean_message(m) for m in messages]
        # Mutated (message DROPPED, not just shortened) only when capping
        # is proven useless for a given attempt - see the no-op check
        # below. Kept separate from the original `messages` so we always
        # know exactly what's still in play.
        working_messages = list(messages)
        token_cap = None  # None = send messages at full length on attempt 1
        oom_step = 0       # how many OOMs hit so far, for walking oom_token_caps
        oom_count = 0
        raw_response = None
        attempt = 0
        prev_msgs_to_send = None  # to detect a cap that changed nothing

        for attempt in range(1, max_retries + 1):
            msgs_to_send = list(working_messages) if token_cap is None else [
                self._truncate_message_tokens(m, token_cap) for m in working_messages
            ]

            # If capping at this level produced the EXACT same text as last
            # attempt, every message was already shorter than the cap - the
            # bottleneck isn't any single message's length (it's most
            # likely the fixed system-prompt overhead), so retrying with an
            # even smaller cap would just reproduce the identical OOM again
            # for free. In that case, actually DROP the longest remaining
            # message instead - that's the only thing left that measurably
            # shrinks the prompt.
            if (token_cap is not None and prev_msgs_to_send is not None
                    and msgs_to_send == prev_msgs_to_send and len(working_messages) > 1):
                lengths = [len(self.tokenizer(m, add_special_tokens=False).input_ids)
                           for m in working_messages]
                drop_idx = max(range(len(working_messages)), key=lambda i: lengths[i])
                print(f"[QwenVerifier]   Capping at {token_cap} token(s) changed nothing for "
                      f"{cluster_label} (every message was already shorter than that) - "
                      f"dropping the longest remaining message instead "
                      f"({len(working_messages)} -> {len(working_messages) - 1}) and retrying "
                      f"(attempt {attempt}/{max_retries})...")
                del working_messages[drop_idx]
                msgs_to_send = list(working_messages) if token_cap is None else [
                    self._truncate_message_tokens(m, token_cap) for m in working_messages
                ]

            prev_msgs_to_send = msgs_to_send

            try:
                raw_response = self.analyze_cluster(msgs_to_send)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    if oom_step < len(oom_token_caps):
                        token_cap = oom_token_caps[oom_step]
                    else:
                        token_cap = max(20, (token_cap or oom_token_caps[-1]) - 50)
                    oom_step += 1
                    print(f"[QwenVerifier]   OUT OF MEMORY on {cluster_label}. "
                          f"Capping each message at {token_cap} token(s) and retrying "
                          f"(attempt {attempt}/{max_retries})...")
                else:
                    print(f"[QwenVerifier]   ERROR generating for {cluster_label}: {e}. "
                          f"Retrying (attempt {attempt}/{max_retries})...")
                continue
            except Exception as e:
                print(f"[QwenVerifier]   ERROR generating for {cluster_label}: {e}. "
                      f"Retrying (attempt {attempt}/{max_retries})...")
                continue

            try:
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                parsed_json = json.loads(json_match.group()) if json_match else json.loads(raw_response)
                return {
                    'parsed_json': parsed_json, 'raw_response': raw_response,
                    'used_token_cap': token_cap, 'attempts': attempt, 'oom_count': oom_count,
                }
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[QwenVerifier]   ERROR parsing model output for {cluster_label} "
                      f"(attempt {attempt}/{max_retries}). Raw output snippet: "
                      f"{raw_response[:100]}...")

        print(f"[QwenVerifier]   FAILED to get valid JSON after {max_retries} attempt(s) "
              f"for {cluster_label} ({oom_count} of them OOM'd). Skipping.")
        return {
            'parsed_json': None, 'raw_response': raw_response,
            'used_token_cap': token_cap, 'attempts': attempt, 'oom_count': oom_count,
        }

    def verify_candidates(self, candidates: list) -> list:
        """
        Receives candidate clusters, samples up to `sample_size` random
        messages from each, sends them to the model, and returns only the
        candidates the model confirmed as a genuine event (with
        event_title / event_summary added).

        Aborts the whole run early only if `self.max_consecutive_oom_before_abort`
        DIFFERENT candidates in a row each completely fail (exhaust their
        own retries with no valid JSON) - one hard candidate no longer
        takes the whole run down with it; see analyze_cluster_with_retry's
        docstring for why that used to happen.
        """
        verified_events = []
        consecutive_candidate_failures = 0
        t_start = time.time()
        print(f"[QwenVerifier] verify_candidates STARTED at {_now()} - "
              f"{len(candidates)} candidate(s)")

        for idx, candidate in enumerate(candidates, 1):
            all_msgs = candidate['messages']
            sample_size = min(self.sample_size, len(all_msgs))
            sampled_for_log = random.sample(all_msgs, sample_size)

            print(f"[QwenVerifier] [{idx}/{len(candidates)}] Analyzing cluster "
                  f"{candidate['cluster_id']} with {sample_size} sampled message(s)...")

            t_cand_start = time.time()
            result = self.analyze_cluster_with_retry(
                sampled_for_log, cluster_label=f"cluster {candidate['cluster_id']}"
            )

            parsed_json = result['parsed_json']
            if parsed_json is None:
                consecutive_candidate_failures += 1
                print(f"[QwenVerifier]   Cluster {candidate['cluster_id']} failed completely "
                      f"({consecutive_candidate_failures} candidate(s) in a row now with no "
                      f"valid result).")
                if consecutive_candidate_failures >= self.max_consecutive_oom_before_abort:
                    print(f"[QwenVerifier]   {consecutive_candidate_failures} candidates in a "
                          f"row failed completely - aborting verify_candidates early to avoid "
                          f"a wasted run (this smells like a GPU/environment problem, not just "
                          f"unlucky clusters).")
                    return verified_events
                continue

            consecutive_candidate_failures = 0
            if parsed_json.get('is_event', False):
                candidate['event_title'] = parsed_json.get('title', 'No title')
                candidate['event_summary'] = parsed_json.get('summary', 'No summary')
                # همان نمونه‌ی ۱۵تایی (یا کمتر) که واقعاً به مدل داده شد را
                # روی خود candidate نگه می‌داریم تا main.py بتواند آن را به
                # عنوان sample_messages در رکورد نهایی ذخیره کند.
                candidate['sample_messages'] = sampled_for_log
                verified_events.append(candidate)
                print(f"[QwenVerifier]   CONFIRMED - title: {candidate['event_title']} "
                      f"({time.time() - t_cand_start:.2f}s, {result['attempts']} attempt(s))")
            else:
                print(f"[QwenVerifier]   REJECTED - not judged a single event "
                      f"({time.time() - t_cand_start:.2f}s, {result['attempts']} attempt(s))")

        elapsed = time.time() - t_start
        print(f"[QwenVerifier] verify_candidates FINISHED at {_now()} - {elapsed:.2f}s | "
              f"{len(verified_events)}/{len(candidates)} confirmed as events")
        return verified_events