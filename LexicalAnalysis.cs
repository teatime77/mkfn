using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace MkFn {
    public enum TokenType {
        Unknown,

        // 識別子
        Identifier,

        // クラス
        Class,

        // 数値
        Number,

        // 記号
        Symbol,

        // 予約語
        ReservedWord,

        // End Of Text
        EOT,

        // 指定なし
        Any,

        Add,
        Sub,
        Mul,
        Div,
        LP,
        RP,
        LB,
        RB,
        LC,
        RC,
    }

    public enum TokenSubType {
        Unknown,
        Integer,
        Float,
        Double,
    }

    public class Token {
        public TokenType Type;
        public TokenSubType SubType;
        public string Text;
        public int LineIndex;
        public int CharPos;

        public Token(TokenType type, TokenSubType sub_type, string text, int line_index, int char_pos) {
            Type = type;
            SubType = sub_type;
            Text = text;
            LineIndex = line_index;
            CharPos = char_pos;
        }
    }

    public partial class MkFn {
        // キーワードの文字列の辞書
        List<string> KeywordMap = new List<string>() {
            "using",
            "public",
            "abstract",
            "class",
            "static",
            "return",
            "foreach",
            "from",
            "in",
            "select",
            "new",
        };

        List<string> SymbolTable = new List<string>() {
            ",",
            ".",
            ";",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "+",
            "-",
            "*",
            "/",
            "%",
            "=",
            ":",
            "<",
            ">",
        };

        List<string> IdList = new List<string>();


        /*
            字句解析をして各文字の字句型の配列を得る。
        */
        public Token[] LexicalAnalysis(string text, int line_idx) {
            List<Token> token_list = new List<Token>();

            // 文字列の長さ
            int text_len = text.Length;

            // 現在の文字位置
            int pos = 0;


            // 文字列の最後までループする。
            while (pos < text_len) {
                
                // 空白をスキップする。
                for ( ; pos < text_len && char.IsWhiteSpace(text[pos]); pos++);
                if(text_len < pos) {
                    // 行末の場合

                    break;
                }

                TokenType token_type = TokenType.Unknown;
                TokenSubType sub_type = TokenSubType.Unknown;

                // 字句の開始位置
                int start_pos = pos;

                // 現在位置の文字
                char ch1 = text[pos];

                // 次の文字の位置。行末の場合は'\0'
                char ch2;

                if (pos + 1 < text.Length) {
                    // 行末でない場合

                    ch2 = text[pos + 1];
                }
                else {
                    // 行末の場合

                    ch2 = '\0';
                }

                if (char.IsLetter(ch1) || ch1 == '_') {
                    // 識別子の最初の文字の場合

                    // 識別子の文字の最後を探す。識別子の文字はユニコードカテゴリーの文字か数字か'_'。
                    for (pos++; pos < text_len && (char.IsLetterOrDigit(text[pos]) || text[pos] == '_'); pos++);

                    // 識別子の文字列
                    string name = text.Substring(start_pos, pos - start_pos);

                    if (KeywordMap.Contains(name)) {
                        // 名前がキーワード辞書にある場合

                        token_type = TokenType.ReservedWord;
                    }
                    else {
                        // 名前がキーワード辞書にない場合

                        if (! IdList.Contains(name)) {

                            IdList.Add(name);
                        }
                        token_type = TokenType.Identifier;
                    }
                }
                else if (char.IsDigit(ch1)) {
                    // 数字の場合

                    token_type = TokenType.Number;

                    // 10進数の終わりを探す。
                    for (; pos < text_len && char.IsDigit(text[pos]); pos++);

                    if (pos < text_len && text[pos] == '.') {
                        // 小数点の場合

                        pos++;

                        // 10進数の終わりを探す。
                        for (; pos < text_len && char.IsDigit(text[pos]); pos++);

                        if (text[pos] == 'f') {

                            pos++;
                            sub_type = TokenSubType.Float;
                        }
                        else {

                            sub_type = TokenSubType.Double;
                        }
                    }
                    else {

                        sub_type = TokenSubType.Integer;
                    }
                }
                else if (ch1 == '/' && ch2 == '/') {
                    // 行コメントの場合

                    break;
                }
                else if (SymbolTable.Contains("" + ch1 + ch2)) {
                    // 2文字の記号の表にある場合

                    token_type = TokenType.Symbol;
                    pos += 2;
                }
                else if (SymbolTable.Contains("" + ch1)) {
                    // 1文字の記号の表にある場合

                    token_type = TokenType.Symbol;
                    pos++;
                }
                else {
                    // 不明の文字の場合

                    token_type = TokenType.Unknown;
                    pos++;
                    Debug.WriteLine("不明 {0}", text.Substring(start_pos, pos - start_pos), "");
//                    throw new Exception();
                }

                // 字句の文字列を得る。
                string s = text.Substring(start_pos, pos - start_pos);

                // トークンを作り、トークンのリストに追加する。
                token_list.Add(new Token(token_type, sub_type, s, line_idx, start_pos));
            }

            // 各文字の字句型の配列を返す。
            return token_list.ToArray();
        }

    }
}

