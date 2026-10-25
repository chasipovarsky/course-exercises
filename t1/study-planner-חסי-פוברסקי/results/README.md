# תוצאות

| קובץ | גרסה | מודל | מקור |
|---|---|---|---|
| `sdk.json` | SDK | `claude-opus-5-5`, effort `medium` | `SDK/main.py --scenario scenario.json --out results/sdk.json` |
| `api_gemini.json` | API (harness ידני) | `gemini-3.5-flash-lite`, `thinking_level=MEDIUM` | `API/main.py --provider gemini --model gemini-3.5-flash-lite --scenario scenario.json --out results/api_gemini.json` |

## למה אין `api.json`

המטלה מבקשת `results/api.json`: ריצה של גרסת ה-API על אותו מודל כמו ה-SDK. הפקודה שבמטלה (`API/main.py --scenario scenario.json --out results/api.json`) מריצה את ה-harness מול Anthropic (`claude-opus-5-5`), וזה דורש `ANTHROPIC_API_KEY`. לא היה לי מפתח של Anthropic, ולכן הקובץ הזה לא נוצר.

במקומו גרסת ה-API נמדדה על Gemini, דרך אותו harness, עם אותו תרחיש ואותן 8 בדיקות. התוצאות ב-`api_gemini.json`, באותו פורמט כמו `sdk.json`. במכוון לא שמרתי אותן בשם `api.json`: השם הזה מבטיח ריצה של Claude, וקובץ Gemini בשם הזה היה מטעה.

מה זה אומר על ההשוואה (מודל שונה, caching שונה, ספירת טוקנים שונה) מוסבר בסעיף 0 של [REPORT.md](../REPORT.md). מימוש ה-Anthropic קיים בקוד כברירת מחדל ונבדק offline. אם יהיה מפתח, הפקודה שבמטלה תיצור את `api.json` בלי שינוי בקוד.

כל השדות לכל תור בשני הקבצים: `input_tokens`, `cache_read_tokens`, `cache_write_tokens`, `output_tokens`, `model_calls`, `tools`, `cost_usd`, `seconds`. בנוסף, ברמת הקובץ: `checks` (תוצאות 8 הבדיקות) ו-`lines_of_code` (שורות לכל חלק ב-harness). ב-`api_gemini.json` יש גם `theoretical_cost_usd`, העלות לפי המחירון בתשלום. העלות בפועל ב-free tier היא 0.
