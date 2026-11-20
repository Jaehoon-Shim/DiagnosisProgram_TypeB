"""3상 전류 기반 전력 시스템 고장진단 GUI."""
from pathlib import Path
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from data import load_rows
from model import diagnose, DiagnosisCancelled

FONT = "맑은 고딕"
APP_TITLE = "3상 전류를 이용한 전력 시스템 고장진단"
COLORS = ("#E75B64", "#D5A117", "#338CD8")
PRESETS = {
    "최소(실험)": dict(reasoning="low", num_predict=512, num_ctx=8192, temperature=0.0, timeout=300),
    "경량(실험)": dict(reasoning="low", num_predict=768, num_ctx=8192, temperature=0.0, timeout=300),
    "신속": dict(reasoning="low", num_predict=1024, num_ctx=8192, temperature=0.0, timeout=600),
    "빠르게": dict(reasoning="low", num_predict=1536, num_ctx=8192, temperature=0.0, timeout=900),
    "균형": dict(reasoning="medium", num_predict=4096, num_ctx=8192, temperature=0.0, timeout=900),
    "깊게": dict(reasoning="high", num_predict=8192, num_ctx=16384, temperature=0.0, timeout=1800),
}


def parse_settings(values):
    """화면 값을 검증해 이번 진단에 고정할 Ollama 설정으로 변환한다."""
    result = dict(values)
    if result.get("reasoning") not in ("low", "medium", "high"):
        raise ValueError("추론 수준은 low, medium, high 중 하나를 선택하세요.")
    for key, low, high, label in (("num_predict", 256, 16384, "추론·생성 한도"),
                                  ("num_ctx", 8192, 65536, "문맥 크기"),
                                  ("timeout", 30, 7200, "행당 최대 대기 시간")):
        try:
            number = int(str(result[key]).replace(",", ""))
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"{label}은 {low:,}~{high:,} 범위의 정수로 입력하세요.") from error
        if not low <= number <= high:
            raise ValueError(f"{label}은 {low:,}~{high:,} 범위로 입력하세요.")
        result[key] = number
    try:
        variation = float(result["temperature"])
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("답변 변동성은 0.0~1.0으로 입력하세요.") from error
    if isinstance(result["temperature"], bool) or not math.isfinite(variation) or not 0 <= variation <= 1:
        raise ValueError("답변 변동성은 0.0~1.0으로 입력하세요.")
    result["temperature"] = variation
    if not isinstance(result.get("user_prompt", ""), str) or len(result.get("user_prompt", "")) > 2000:
        raise ValueError("추가 진단 지시는 2,000자 이내로 입력하세요.")
    result["user_prompt"] = result.get("user_prompt", "")
    return result


class App(tk.Tk):
    # 1. 화면 구성과 입력 파일 선택
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{min(1180, self.winfo_screenwidth()-60)}x{min(840, self.winfo_screenheight()-90)}")
        self.minsize(1040, 740)
        self.configure(bg="#F1F5FA")
        self.option_add("*Font", (FONT, 10))
        self.rows = []
        self.index = 0
        self.busy = False
        self.closing = False
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.file_text = tk.StringVar(value="CSV 또는 Excel 파일을 선택하세요.")
        self.row_text = tk.StringVar()
        self.model_name = tk.StringVar(value="gpt-oss:20b")
        self.preset = tk.StringVar(value="신속")
        self.reasoning = tk.StringVar(value="low")
        self.output_tokens = tk.StringVar(value="1024")
        self.context_tokens = tk.StringVar(value="8192")
        self.variation = tk.StringVar(value="0.0")
        self.request_timeout = tk.StringVar(value="600")
        self.additional_prompt = tk.StringVar(value="")
        self.settings_hint = tk.StringVar()
        self.used_settings = tk.StringVar(value="모델의 최종 판단")
        self.header = tk.BooleanVar(value=False)
        self.summary = tk.StringVar(value="데이터를 선택하세요")
        self.uncertainty = tk.StringVar()
        self.status = tk.StringVar(value="Ollama가 실행 중이고 gpt-oss:20b가 설치되어 있어야 합니다.")
        self._build()
        self.update_settings_hint()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = self.after(100, self.poll)
        folder = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
        beside = folder / "data_200_200_200.csv"
        if beside.is_file():
            self.after(100, lambda: self.open_file(beside))

    def _build(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=(14, 8), font=(FONT, 10, "bold"),
                        background="#FFFFFF", foreground="#233C58", bordercolor="#CBD7E6")
        style.map("TButton", background=[("active", "#E8F0FB")],
                  foreground=[("disabled", "#8B9AAD")])
        style.configure("Primary.TButton", background="#2464B5", foreground="white", bordercolor="#2464B5")
        style.map("Primary.TButton", background=[("disabled", "#ADC3DF"), ("active", "#184C91")],
                  foreground=[("disabled", "#F5F8FC"), ("active", "white")])
        style.configure("Stop.TButton", foreground="#A62F43", bordercolor="#D9C8CD")
        style.configure("TCheckbutton", background="#F1F5FA", foreground="#344D68")
        style.configure("TCombobox", padding=4)
        title = tk.Frame(self, bg="#112B46")
        title.pack(fill="x")
        tk.Label(title, text=APP_TITLE, bg="#112B46", fg="white",
                 font=(FONT, 20, "bold")).pack(side="left", padx=23, pady=18)
        tk.Label(title, text="LOCAL AI", bg="#112B46", fg="#72D9EB",
                 font=("Segoe UI", 10, "bold")).pack(side="right", padx=23)
        tk.Frame(self, bg="#2B83CB", height=3).pack(fill="x")
        body = tk.Frame(self, bg="#F1F5FA")
        body.pack(fill="both", expand=True, padx=22, pady=16)
        tools = tk.Frame(body, bg="#F1F5FA")
        tools.pack(fill="x")
        self.open_button = ttk.Button(tools, text="데이터 열기", command=self.open_file)
        self.open_button.pack(side="left")
        self.header_check = ttk.Checkbutton(tools, text="첫 행은 열 이름", variable=self.header)
        self.header_check.pack(side="left", padx=12)
        tk.Label(tools, text="행 선택", bg="#F1F5FA", fg="#233C58", font=(FONT, 10, "bold")).pack(side="left", padx=(15, 5))
        self.row_box = ttk.Combobox(tools, textvariable=self.row_text, width=8, state="readonly")
        self.row_box.pack(side="left")
        self.row_box.bind("<<ComboboxSelected>>", self.select_row)
        tk.Label(body, textvariable=self.file_text, anchor="w", bg="#F1F5FA", fg="#465F7B",
                 wraplength=1050).pack(fill="x", pady=(8, 12))
        settings = tk.Frame(body, bg="#F1F5FA")
        settings.pack(fill="x")
        tk.Label(settings, text="모델", bg="#F1F5FA", fg="#233C58", font=(FONT, 10, "bold")).pack(side="left")
        self.model_entry = ttk.Entry(settings, textvariable=self.model_name, width=21)
        self.model_entry.pack(side="left", padx=(6, 20))
        tk.Label(settings, text="분석 모드", bg="#F1F5FA", fg="#233C58", font=(FONT, 10, "bold")).pack(side="left")
        self.preset_box = ttk.Combobox(settings, textvariable=self.preset, width=12,
                                      values=(*PRESETS, "사용자 설정"), state="readonly")
        self.preset_box.pack(side="left", padx=(7, 12))
        self.preset_box.bind("<<ComboboxSelected>>", self.apply_preset)
        self.tuning_button = ttk.Button(settings, text="Ollama 상세 설정", command=self.open_settings)
        self.tuning_button.pack(side="left")
        tk.Label(body, textvariable=self.settings_hint, bg="#F1F5FA", fg="#465F7B", anchor="w",
                 wraplength=1090, font=(FONT, 9)).pack(fill="x", pady=(7, 0))
        actions = tk.Frame(body, bg="#F1F5FA")
        actions.pack(fill="x", pady=11)
        self.one_button = ttk.Button(actions, text="현재 행 진단", style="Primary.TButton", command=lambda: self.start(False))
        self.all_button = ttk.Button(actions, text="전체 순차 진단", command=lambda: self.start(True))
        self.stop_button = ttk.Button(actions, text="중지", style="Stop.TButton", command=self.stop, state="disabled")
        for button in (self.one_button, self.all_button, self.stop_button):
            button.pack(side="left", padx=(0, 8))
        self.plot = tk.Canvas(body, bg="white", height=280, highlightthickness=1,
                              highlightbackground="#BACBDD")
        self.plot.pack(fill="both", expand=True)
        self.plot.bind("<Configure>", lambda _: self.draw_waveform())
        results = tk.Frame(body, bg="#F1F5FA")
        results.pack(fill="x", pady=(16, 0))
        results.grid_columnconfigure(0, minsize=290)
        results.grid_columnconfigure(1, weight=1)
        diagnosis_frame = tk.Frame(results, bg="white", highlightbackground="#AABED6", highlightthickness=1)
        diagnosis_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        tk.Frame(diagnosis_frame, bg="#2464B5", width=5).pack(side="left", fill="y")
        diagnosis_body = tk.Frame(diagnosis_frame, bg="white", padx=16, pady=12)
        diagnosis_body.pack(side="left", fill="both", expand=True)
        tk.Label(diagnosis_body, text="진단", bg="white", fg="#102C4D", anchor="w",
                 font=(FONT, 14, "bold")).pack(fill="x", pady=(0, 10))
        self.summary_label = tk.Label(diagnosis_body, textvariable=self.summary, bg="white", fg="#123D70",
                                      anchor="w", justify="left", wraplength=255, font=(FONT, 21, "bold"))
        self.summary_label.pack(fill="both", expand=True)
        tk.Label(diagnosis_body, textvariable=self.used_settings, bg="white", fg="#465F7B", anchor="w",
                 justify="left", wraplength=255, font=(FONT, 9)).pack(fill="x", pady=(6, 0))
        explanation = tk.Frame(results, bg="white", highlightbackground="#AABED6", highlightthickness=1,
                               padx=16, pady=12)
        explanation.grid(row=0, column=1, sticky="nsew")
        tk.Label(explanation, text="모델이 설명한 이유", bg="white", fg="#102C4D", anchor="w",
                 font=(FONT, 14, "bold")).pack(fill="x", pady=(0, 7))
        reason_frame = tk.Frame(explanation, bg="white")
        reason_frame.pack(fill="both", expand=True)
        self.reason = tk.Text(reason_frame, height=4, wrap="word", font=(FONT, 12),
                              fg="#172F4D", bg="white", selectbackground="#CFE3FC", relief="flat",
                              padx=0, pady=2, state="disabled")
        scroll = ttk.Scrollbar(reason_frame, command=self.reason.yview)
        self.reason.configure(yscrollcommand=scroll.set)
        self.reason.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        tk.Label(body, textvariable=self.uncertainty, bg="#F1F5FA", fg="#465F7B",
                 wraplength=1080, justify="left", anchor="w").pack(fill="x", pady=8)
        tk.Label(body, textvariable=self.status, bg="#DDE8F5", fg="#23486E", font=(FONT, 10, "bold"),
                 wraplength=1080, justify="left", anchor="w", padx=12, pady=9).pack(fill="x")

    def read_inference_settings(self):
        return parse_settings(dict(reasoning=self.reasoning.get(), num_predict=self.output_tokens.get(),
                                   num_ctx=self.context_tokens.get(), temperature=self.variation.get(),
                                   timeout=self.request_timeout.get(), user_prompt=self.additional_prompt.get()))

    def apply_preset(self, _event=None):
        if self.busy:
            return
        profile = PRESETS.get(self.preset.get())
        if profile:
            for variable, key in ((self.reasoning, "reasoning"), (self.output_tokens, "num_predict"),
                                  (self.context_tokens, "num_ctx"), (self.variation, "temperature"),
                                  (self.request_timeout, "timeout")):
                variable.set(str(profile[key]))
        self.update_settings_hint()

    def update_settings_hint(self):
        caution = ("실험 모드: 답변이 잘릴 수 있으며, 잘린 답변은 진단으로 표시하지 않습니다."
                   if self.preset.get() in ("최소(실험)", "경량(실험)") else
                   "추론 low가 gpt-oss의 최저 수준입니다. 속도와 정확도는 입력에 따라 달라집니다.")
        self.settings_hint.set(f"현재 설정: {self.preset.get()} / 추론 {self.reasoning.get()} / 한도 {self.output_tokens.get()}토큰 / "
                               f"최대 {self.request_timeout.get()}초·행. {caution}")

    def open_settings(self):
        if self.busy:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Ollama 상세 설정")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(bg="#F1F5FA")
        frame = tk.Frame(dialog, bg="#F1F5FA", padx=22, pady=18)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Ollama 추론 설정", bg="#F1F5FA", fg="#102C4D",
                 font=(FONT, 16, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
        staged = {key: tk.StringVar(dialog, value=value) for key, value in
                  dict(reasoning=self.reasoning.get(), num_predict=self.output_tokens.get(),
                       num_ctx=self.context_tokens.get(), temperature=self.variation.get(),
                       timeout=self.request_timeout.get()).items()}
        fields = [("추론 수준", "reasoning", ("low", "medium", "high"), "깊을수록 추론 시간이 늘 수 있습니다. gpt-oss 기준입니다."),
                  ("추론·생성 한도 (토큰)", "num_predict", ("256", "512", "768", "1024", "1536", "2048", "4096", "8192", "16384"), "추론과 답변의 공통 상한입니다. 너무 낮으면 답변이 잘립니다."),
                  ("문맥 크기 (토큰)", "num_ctx", ("8192", "16384", "32768", "65536"), "입력과 생성에 쓸 공간입니다. 크게 하면 메모리를 더 씁니다."),
                  ("답변 변동성 (0~1)", "temperature", ("0.0", "0.1", "0.2", "0.5", "1.0"), "0은 답변 변동성을 줄입니다. 정확도 조절 값은 아닙니다."),
                  ("행당 최대 대기 (초)", "timeout", ("30", "60", "120", "300", "600", "900", "1800", "3600", "7200"), "전체 제한 시간이며, 낮춘다고 계산 자체가 빨라지지는 않습니다.")]
        for row, (label, key, choices, help_text) in enumerate(fields, start=1):
            tk.Label(frame, text=label, bg="#F1F5FA", fg="#233C58", font=(FONT, 10, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
            ttk.Combobox(frame, textvariable=staged[key], values=choices, width=12,
                         state="readonly" if key == "reasoning" else "normal").grid(row=row, column=1, sticky="w")
            tk.Label(frame, text=help_text, bg="#F1F5FA", fg="#465F7B", wraplength=380,
                     justify="left").grid(row=row, column=2, sticky="w", padx=(14, 0))
        tk.Label(frame, text="추가 진단 지시·측정 조건 (선택, 최대 2,000자)", bg="#F1F5FA", fg="#233C58",
                 font=(FONT, 10, "bold")).grid(row=6, column=0, columnspan=3, sticky="w", pady=(16, 6))
        prompt = tk.Text(frame, height=4, width=81, wrap="word", font=(FONT, 11),
                         bg="white", fg="#172F4D", relief="solid", borderwidth=1)
        prompt.grid(row=7, column=0, columnspan=3, sticky="ew")
        prompt.insert("1.0", self.additional_prompt.get())
        tk.Label(frame, text="전류 단위·샘플링 주파수·회로 조건처럼 알고 있는 사실을 적으세요. 원본 600개와 함께 전달합니다.",
                 bg="#F1F5FA", fg="#465F7B", wraplength=790, justify="left").grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 12))

        def apply():
            try:
                values = parse_settings({**{key: value.get() for key, value in staged.items()},
                                         "user_prompt": prompt.get("1.0", "end-1c")})
            except ValueError as error:
                messagebox.showerror("설정 확인", str(error), parent=dialog)
                return
            for variable, key in ((self.reasoning, "reasoning"), (self.output_tokens, "num_predict"),
                                  (self.context_tokens, "num_ctx"), (self.variation, "temperature"),
                                  (self.request_timeout, "timeout"), (self.additional_prompt, "user_prompt")):
                variable.set(str(values[key]))
            self.preset.set("사용자 설정")
            self.update_settings_hint()
            dialog.destroy()

        buttons = tk.Frame(frame, bg="#F1F5FA")
        buttons.grid(row=9, column=0, columnspan=3, sticky="e")
        ttk.Button(buttons, text="취소", command=dialog.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(buttons, text="적용", style="Primary.TButton", command=apply).pack(side="right")
        dialog.grab_set()

    def open_file(self, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(filetypes=[("CSV / Excel", "*.csv *.xlsx *.xlsm")])
        if not path:
            return
        try:
            rows = load_rows(path, skip_header=self.header.get())
            if not rows:
                raise ValueError("숫자 600개가 있는 데이터 행을 찾지 못했습니다.")
        except Exception as error:
            messagebox.showerror("데이터 확인", str(error))
            return
        self.rows = rows
        self.row_box.configure(values=[str(number) for number, _ in rows])
        self.file_text.set(f"{Path(path).name} · {len(rows)}개 행 · ia 200 + ib 200 + ic 200 (앞 600열)")
        self.show_row(0)
        self.status.set("파형을 확인한 후 현재 행 또는 전체 순차 진단을 실행하세요.")

    def select_row(self, _event=None):
        if not self.busy and self.rows:
            self.show_row(self.row_box.current())

    def show_row(self, index):
        self.index = index
        self.row_text.set(str(self.rows[index][0]))
        self.summary.set("미진단")
        self.set_reason("")
        self.uncertainty.set("")
        self.used_settings.set("모델의 최종 판단")
        self.draw_waveform()

    def set_reason(self, value):
        self.reason.configure(state="normal")
        self.reason.delete("1.0", "end")
        self.reason.insert("1.0", value)
        self.reason.configure(state="disabled")

    # 2. 원본 600개를 파형으로 표시한다. 진단 규칙은 계산하지 않는다.
    def draw_waveform(self):
        self.plot.delete("all")
        width, height = self.plot.winfo_width(), self.plot.winfo_height()
        if width < 100 or height < 100 or not self.rows:
            return
        row, currents = self.rows[self.index]
        left, right, top, bottom = 55, width - 22, 32, height - 38
        peak = max(abs(float(v)) for phase in currents for v in phase) or 1.0
        for k in range(5):
            y = top + k * (bottom - top) / 4
            self.plot.create_line(left, y, right, y, fill="#E5EBF2")
            self.plot.create_text(left - 7, y, text=f"{(1-k/2)*peak:.3g}", anchor="e", fill="#64758A")
        for n in (0, 50, 100, 150, 199):
            x = left + n * (right - left) / 199
            self.plot.create_line(x, top, x, bottom, fill="#EFF3F7")
            self.plot.create_text(x, bottom + 16, text=str(n), fill="#64758A")
        for phase, color in enumerate(COLORS):
            points = []
            for n, value in enumerate(currents[phase]):
                points.extend((left + n * (right-left)/199,
                               (top+bottom)/2 - (float(value)/peak) * (bottom-top)/2))
            self.plot.create_line(*points, fill=color, width=2, smooth=False)
            self.plot.create_text(right-180+phase*70, 15, text=("ia", "ib", "ic")[phase], fill=color)
        self.plot.create_text(left, 15, text=f"입력 행 {row}", anchor="w", fill="#203D5C")

    # 3. 모델 호출은 별도 스레드에서 한 행씩 실행한다.
    def start(self, batch):
        if self.busy:
            return
        if not self.rows:
            messagebox.showinfo("데이터 선택", "먼저 데이터 파일을 열어주세요.")
            return
        model = self.model_name.get().strip()
        if not model:
            messagebox.showinfo("모델 이름", "설치한 Ollama 모델 이름을 입력하세요.")
            return
        try:
            settings = self.read_inference_settings()
        except ValueError as error:
            messagebox.showerror("Ollama 설정 확인", str(error))
            return
        self.cancel.clear()
        self.set_busy(True)
        indices = range(len(self.rows)) if batch else [self.index]
        threading.Thread(target=self.worker,
                         args=(indices, model, settings), daemon=True).start()

    def worker(self, indices, model, settings):
        in_flight = False
        try:
            for index in indices:
                if self.cancel.is_set():
                    raise DiagnosisCancelled("중지했습니다.")
                row, currents = self.rows[index]
                in_flight = True
                self.events.put(("row", index))
                result = diagnose(currents, model=model, current_direction="inverter_to_load",
                                  cancel_event=self.cancel,
                                  progress=lambda text, r=row: self.events.put(("status", f"{r}행 · {text}")), **settings)
                if self.cancel.is_set():
                    raise DiagnosisCancelled("중지했습니다.")
                self.events.put(("result", (index, result, settings)))
                in_flight = False
            self.events.put(("status", "진단 완료"))
        except DiagnosisCancelled:
            if in_flight:
                self.events.put(("cancelled", None))
            self.events.put(("status", "중지 완료 · 다음 행을 시작하지 않습니다."))
        except Exception as error:
            self.events.put(("error", str(error)))
        finally:
            self.events.put(("done", None))

    # 4. 메인 스레드에서만 위젯을 갱신한다.
    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "row":
                    self.show_row(value)
                    self.summary.set("모델 진단 중…")
                elif kind == "result":
                    index, result, settings = value
                    self.summary.set(result["label"])
                    self.set_reason(result["reason"])
                    self.uncertainty.set(result["uncertainty"])
                    self.used_settings.set(f"사용 설정: {settings['reasoning']} / {settings['num_predict']:,}토큰")
                elif kind == "status":
                    self.status.set(value)
                elif kind == "error":
                    self.summary.set("진단 실패")
                    self.set_reason(value)
                    self.status.set("설정과 오류 내용을 확인한 뒤 현재 행을 다시 진단하세요.")
                elif kind == "cancelled":
                    self.summary.set("진단 중지 · 미진단")
                    self.set_reason("")
                    self.uncertainty.set("")
                elif kind == "done":
                    self.set_busy(False)
        except queue.Empty:
            pass
        if not self.closing:
            self.poll_id = self.after(100, self.poll)

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.open_button, self.header_check, self.model_entry, self.one_button,
                       self.all_button, self.tuning_button):
            widget.configure(state="disabled" if busy else "normal")
        self.row_box.configure(state="disabled" if busy else "readonly")
        self.preset_box.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def stop(self):
        self.cancel.set()
        self.stop_button.configure(state="disabled")
        self.status.set("중지 요청 · 현재 모델 연결을 종료합니다.")

    def close(self):
        self.closing = True
        self.cancel.set()
        self.after_cancel(self.poll_id)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
