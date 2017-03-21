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

        // 行コメント
        LineComment,

        // ブロックコメント
        BlockComment,

        // 改行
        NewLine,

        // 文字列
        String,

        // 文字
        Character,

        // 不正
        Illegal,
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

        public override string ToString() {
            return Text;
        }
    }

    public partial class MkFn {
        // キーワードの文字列の辞書
        List<string> KeywordMap = new List<string>() {
/*
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
*/
            "abstract",
            "as",
            "base",
            "break",
            "case",
            "catch",
            "checked",
            "class",
            "const",
            "continue",
            "default",
            "delegate",
            "do",
            "else",
            "enum",
            "event",
            "explicit",
            "extern",
            "finally",
            "fixed",
            "for",
            "foreach",
            "goto",
            "if",
            "implicit",
            "in",
            "interface",
            "internal",
            "is",
            "lock",
            "namespace",
            "new",
            "operator",
            "out",
            "override",
            "params",
            "private",
            "protected",
            "public",
            "readonly",
            "ref",
            "return",
            "sealed",
            "sizeof",
            "stackalloc",
            "static",
            "struct",
            "switch",
            "throw",
            "try",
            "typeof",
            "unchecked",
            "unsafe",
            "using",
            "virtual",
            "volatile",
            "while",

//          "add",
            "alias",
            "ascending",
            "async",
            "await",
            "descending",
            "dynamic",
            "from",
            "get",
            "global",
            "group",
            "into",
            "join",
            "let",
            "orderby",
            "partial",
//          "remove",
            "select",
            "set",
//          "value",
            "var",
            "where",
            "yield",
        };

        List<string> SystemValueTable = new List<string>() {
            "false",
            "null",
            "this",
            "true",
        };

        List<string> SystemClassTable = new List<string>() {
            "bool",
            "byte",
            "char",
            "decimal",
            "double",
            "float",
            "int",
            "long",
            "object",
            "sbyte",
            "short",
            "string",
            "uint",
            "ulong",
            "ushort",
            "void",
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

            "&&",
            "||",

            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "!=",

            "++",
            "--",

            "!",
            "&",
            "|",
            "?",
        };

        List<string> IdList = new List<string>();


        /*
            字句解析をして各文字の字句型の配列を得ます。
        */
        public Token[] LexicalAnalysis(string text, int line_idx) {
            List<Token> token_list = new List<Token>();

            // 現在の文字位置
            int pos = 0;

            // 行の先頭位置
            int line_top = 0;

            // 文字列の最後までループします。
            while (pos < text.Length) {

                // 字句の開始位置
                int start_pos = pos;

                TokenType token_type = TokenType.Unknown;
                TokenSubType sub_type = TokenSubType.Unknown;
                
                // 改行以外の空白をスキップします。
                for ( ; pos < text.Length && text[pos] != '\r' && text[pos] != '\n' && char.IsWhiteSpace(text[pos]); pos++);

                if (text.Length <= pos) {
                    // テキストの終わりの場合

                    break;
                }
                start_pos = pos;

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

                if (ch1 == '\r' || ch1 == '\n') {
                    // CRかLFの場合

                    if (ch1 == '\r' && ch2 == '\n') {
                        // CR+LFの場合

                        pos += 2;
                    }
                    else {
                        // CR+LFでない場合

                        pos++;
                    }
                    line_top = pos;

                    line_idx++;
                    token_type = TokenType.NewLine;
                }
                else if (char.IsLetter(ch1) || ch1 == '_') {
                    // 識別子の最初の文字の場合

                    // 識別子の文字の最後を探します。識別子の文字はユニコードカテゴリーの文字か数字か'_'。
                    for (pos++; pos < text.Length && (char.IsLetterOrDigit(text[pos]) || text[pos] == '_'); pos++);

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

                    // 10進数の終わりを探します。
                    for (; pos < text.Length && char.IsDigit(text[pos]); pos++);

                    if (pos < text.Length && text[pos] == '.') {
                        // 小数点の場合

                        pos++;

                        // 10進数の終わりを探します。
                        for (; pos < text.Length && char.IsDigit(text[pos]); pos++);

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

                    // 行末を探します。
                    for (; pos < text.Length && text[pos] != '\r' && text[pos] != '\n'; pos++) ;

                    token_type = TokenType.LineComment;
                }
                else if (ch1 == '/' && ch2 == '*') {
                    // ブロックコメントの場合

                    // ブロックコメントの終わりを探します。
                    pos = text.IndexOf("*/", start_pos + 2);
                    if (pos == -1) {
                        // ブロックコメントの終わりがない場合。

                        pos = text.Length;

                        token_type = TokenType.Illegal;
                    }
                    else {
                        // ブロックコメントの終わりがある場合。

                        pos += 2;
                        token_type = TokenType.BlockComment;
                    }
                }
                else if (ch1 == '@' && ch2 == '\"') {
                    // 逐語的文字列の場合

                    pos += 2;

                    // 逐語的文字列の終わりを探します。
                    while (true) {

                        // 逐語的文字列の終わりを探します。
                        pos = text.IndexOf('\"', pos);
                        if (pos == -1) {
                            // 逐語的文字列の終わりがない場合。

                            pos = text.Length;

                            token_type = TokenType.Illegal;
                            goto add_token;
                        }

                        if (pos + 1 < text.Length && text[pos + 1] == '\"') {
                            // 次の文字が"の場合

                            continue;
                        }

                        pos++;
                        break;
                    }

                    token_type = TokenType.String;
                }
                else if (ch1 == '\"') {
                    // 文字列の場合

                    pos++;

                    // 文字列の終わりを探します。
                    while(true) {
                        if (text.Length <= pos) {

                            token_type = TokenType.Illegal;
                            break;
                        }
                        else {

                            if (text[pos] == '"') {

                                pos++;
                                token_type = TokenType.String;
                                break;
                            }
                            else if (text[pos] == '\\') {

                                pos += 2;
                            }
                            else {
                                pos++;
                            }
                        }
                    }
                }
                else if (ch1 == '\'') {
                    // 文字の場合

                    if(ch2 == '\\') {

                        pos += 3;
                    }
                    else {

                        pos += 2;
                    }
                    if(pos < text.Length && text[pos] == '\'') {
                        // 文字の終わりがある場合

                        token_type = TokenType.Character;
                        pos++;
                    }
                    else {
                        // 文字の終わりがない場合

                        token_type = TokenType.Illegal;
                    }
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
                add_token:

                // 字句の文字列を得ます。
                string s = text.Substring(start_pos, pos - start_pos);

                // トークンを作り、トークンのリストに追加します。
                token_list.Add(new Token(token_type, sub_type, s, line_idx, start_pos - line_top));

                if(token_type == TokenType.Illegal) {

                    Debug.WriteLine("不正 {0} ^ {1}", text.Substring(line_top, start_pos - line_top), s, "");
                }
            }

            // 各文字の字句型の配列を返します。
            return token_list.ToArray();
        }

        public Token[] RemoveComment(Token[] tokens) {
            return tokens.Where(x => x.Type != TokenType.LineComment && x.Type != TokenType.BlockComment).ToArray();
        }
    }
}
