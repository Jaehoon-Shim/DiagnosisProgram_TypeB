"""Excel/CSV의 한 행에서 ia, ib, ic 각각 200개 전류를 읽는다."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from openpyxl import load_workbook


# 1. CSV 문자 인코딩: UTF-8을 먼저 시도하고, 국내 Excel의 CP949도 허용한다.
def _csv_rows(path: Path):
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="cp949")
        except UnicodeDecodeError as error:
            raise ValueError("CSV를 UTF-8 또는 CP949 형식으로 다시 저장해 주세요.") from error
    yield from csv.reader(io.StringIO(text, newline=""))


# 2. Excel은 첫 시트를 읽기 전용으로 연다. 수식은 저장된 계산 결과를 읽는다.
def _excel_rows(path: Path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        yield from workbook.worksheets[0].iter_rows(max_col=600, values_only=True)
    finally:
        workbook.close()


# 3. 선택한 600개 셀을 확인하고 [ia 200개, ib 200개, ic 200개]로 나눈다.
def _to_currents(row, row_number: int) -> np.ndarray | None:
    selected = list(row[:600])
    if all(value is None or isinstance(value, str) and not value.strip()
           for value in selected):
        return None
    selected.extend([None] * (600 - len(selected)))
    values = []
    for column, value in enumerate(selected, start=1):
        location = f"{row_number}행 {column}열"
        if value is None or isinstance(value, str) and not value.strip():
            raise ValueError(f"{location}: 값이 비어 있습니다. 행마다 숫자 600개가 필요합니다.")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{location}: 논리값(True/False)은 전류 숫자로 사용할 수 없습니다.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{location}: 숫자가 아닌 값 {value!r}이 있습니다.") from error
        if not np.isfinite(number):
            raise ValueError(f"{location}: NaN이나 무한대 대신 유한한 숫자를 넣어 주세요.")
        values.append(number)
    return np.asarray(values, dtype=np.float64).reshape(3, 200)


# 4. 원본 행 번호를 유지한다. 601번째 이후 열과 정답 라벨은 사용하지 않는다.
def load_rows(path, skip_header: bool = False) -> list[tuple[int, np.ndarray]]:
    """첫 600열을 읽어 [(원본 1기반 행 번호, float64[3, 200]), ...]로 반환.

    skip_header=True이면 첫 행 하나를 제외한다. 선택한 600개 셀이 모두 빈
    행은 건너뛴다. 잘못된 셀이 있으면 해당 행·열을 포함한 ValueError를 낸다.
    원본을 저장하거나 데이터를 학습에 이용하지 않는다.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"파일을 찾을 수 없습니다: {source}")
    if source.suffix.lower() not in {".csv", ".xlsx", ".xlsm"}:
        raise ValueError("CSV, XLSX, XLSM 파일을 선택해 주세요. XLS는 XLSX로 저장해 주세요.")
    reader = _csv_rows(source) if source.suffix.lower() == ".csv" else _excel_rows(source)
    result = []
    try:
        for row_number, row in enumerate(reader, start=1):
            if skip_header and row_number == 1:
                continue
            currents = _to_currents(row, row_number)
            if currents is not None:
                result.append((row_number, currents))
    except (OSError, BadZipFile) as error:
        raise ValueError(f"파일을 읽을 수 없습니다. 형식과 열림 상태를 확인해 주세요: {source.name}") from error
    finally:
        reader.close()
    return result
