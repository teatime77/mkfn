using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MkFn {
    class Program {
        static void Main(string[] args) {
            new MkFn().Run();
        }
    }

    public partial class MkFn {
        public void Run() {
            string source_dir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location))));
            List<string> file_list = Directory.GetFiles(source_dir, "*.cs").ToList();

            file_list.Add(source_dir + @"\sample\NeuralNetwork.cs");

            StringWriter sw = new StringWriter();
            sw.WriteLine("@startuml");

            Dictionary<string, Class> class_dic = new Dictionary<string, Class>();

            Dictionary<string, List<Token[]>> file_lines = new Dictionary<string, List<Token[]>>();

            foreach (string file_path in file_list) {
                Debug.WriteLine(file_path);

                string text = File.ReadAllText(file_path, Encoding.UTF8);
                Token[] all_tokens = LexicalAnalysis(text, 0);

                string s = string.Join("", from x in all_tokens select x.Text);
                //if(text != s) {
                //    Debug.WriteLine("");
                //}

                List<Token[]> lines = new List<Token[]>();
                List<Token> tmp = new List<Token>();
                for(int i = 0; i < all_tokens.Length; i++) {
                    Token tkn = all_tokens[i];
                    if(tkn.Type == TokenType.NewLine) {
                        lines.Add(tmp.ToArray());
                        tmp.Clear();
                    }
                    else {
                        tmp.Add(tkn);
                    }
                }

                file_lines.Add(file_path, lines);

                Class cls = null;
                Method fnc = null;

                foreach (Token[] tokens in lines) {

                    if(tokens.Length == 0) {
                        // 空行の場合

                        if(fnc != null) {
                            fnc.Lines.Add(tokens);
                        }
                        continue;
                    }

                    Token first_token = tokens[0];
                    Token last_token = tokens[tokens.Length - 1];
                    if(last_token.Text == "{") {

                        if(first_token.Text == "") {

                        }
                    }
                    else if (last_token.Text == ";") {

                    }
                    else if (last_token.Text == ":") {

                    }
                    var vv = Range(tokens.Length).Where(i => tokens[i].Text == "class");
                    if (vv.Any()) {
                        int idx = vv.First();

                        string class_name = tokens[idx + 1].Text;
                        if (!class_dic.TryGetValue(class_name, out cls)) {

                            cls = new Class(class_name);
                            class_dic.Add(class_name, cls);
                        }

                        if(idx + 3 < tokens.Length && tokens[idx + 2].Text == ":") {

                            cls.SuperClassName = tokens[idx + 3].Text;
                        }
                    }
                    else if (first_token.CharPos == 8) {
                        if (last_token.Text == "{") {

                            var v = from i in Range(tokens.Length) where tokens[i].Text == "(" select i;
                            if (v.Any()) {

                                fnc = new Method(tokens[v.First() - 1].Text);
                                fnc.Lines.Add(tokens);
                                cls.Methods.Add(fnc);
                                Debug.WriteLine("{0}.{1}", cls.Name, fnc.Name);
                            }
                        }
                        else if(first_token.Text == "}"){
                            if(fnc != null) {

                                fnc.Lines.Add(tokens);
                                fnc = null;
                            }
                        }
                    }
                    else if (fnc != null && 8 < first_token.CharPos) {

                        fnc.Lines.Add(tokens);
                    }
                }
            }

            class_dic.Add("List", new Class("List"));
            class_dic.Add("Dictionary", new Class("Dictionary"));

            var ids = from lines in file_lines.Values from tokens in lines from token in tokens where token.Type == TokenType.Identifier select token;
            foreach (Token id in ids) {
                if (class_dic.ContainsKey(id.Text)) {
                    id.Type = TokenType.Class;
                }
            }

            foreach (Class cls in class_dic.Values) {
                sw.WriteLine("class {0} {{", cls.Name);
                sw.WriteLine("}");
                if (cls.SuperClassName != null) {

                    sw.WriteLine("{0} <|-- {1}", cls.SuperClassName, cls.Name);
                }
            }

            sw.WriteLine("@enduml");
            File.WriteAllText("class.txt", sw.ToString(), Encoding.UTF8);

            WriteHTML(file_lines);


            List<string> src_list = Directory.GetFiles(@"C:\usr\prj\mkfn\doc\src", "*.html").ToList();
            string dst_dir = @"C:\usr\prj\mkfn\doc\";
            foreach(string src_path in src_list) {
                string text = File.ReadAllText(src_path, Encoding.UTF8);

                while (true) {
                    int pos = text.IndexOf("<!--#");
                    if(pos == -1) {
                        break;
                    }
                    int st =pos + 5;
                    int i = text.IndexOf(".", st);
                    int j = text.IndexOf("-->", st);

                    string class_name = text.Substring(st, i - st);
                    string fnc_name = text.Substring(i + 1, j - (i + 1));

                    Method fnc = class_dic[class_name].Methods.Where(x => x.Name == fnc_name).First();

                    text = text.Substring(0, pos) + "\r\n" + Code(fnc.Lines) + text.Substring(j + 3);
                }

                string dst_path = dst_dir + Path.GetFileName(src_path);
                File.WriteAllText(dst_path, text, Encoding.UTF8);
            }

        }

        public int[] Range(int n) {
            return Enumerable.Range(0, n).ToArray();
        }

        string Code(List<Token[]> lines) {
            StringWriter sw = new StringWriter();

            sw.WriteLine("<pre>");
            foreach (Token[] tokens in lines) {
                int pos = 0;

                foreach (Token token in tokens) {
                    if (pos < token.CharPos) {
                        sw.Write(new string(' ', token.CharPos - pos));
                    }

                    string text = token.Text.Replace("<", "&lt;").Replace(">", "&gt;");
                    string color = "";
                    switch (token.Type) {
                    case TokenType.BlockComment:
                    case TokenType.LineComment:
                        color = "green";//comment
                        break;

                    case TokenType.ReservedWord:
                        color = "blue";//reserved
                        break;

                    case TokenType.String:
                    case TokenType.Character:
                        color = "red";//string
                        break;

                    case TokenType.Class:
                        color = "steelblue";//class
                        break;

                    case TokenType.Identifier:
                        if (SystemValueTable.Contains(text) || SystemClassTable.Contains(text)) {
                            color = "blue";//reserved
                        }
                        break;
                    }

                    if (color == "") {
                        sw.Write(token.Text);
                    }
                    else {

                        sw.Write("<span style='color:{0}'>{1}</span>", color, text);
                    }

                    pos = token.CharPos + token.Text.Length;
                }

                sw.WriteLine("");
            }
            sw.WriteLine("</pre>");

            return sw.ToString();
        }

        void WriteHTML(Dictionary<string, List<Token[]>> file_lines) {
            string header = @"<!DOCTYPE html>

<html lang=""en"" xmlns=""http://www.w3.org/1999/xhtml"">
<head>
    <meta charset=""utf-8"" />
    <title>mkfn - ディープラーニングのフレームワーク</title>
    <script type=""text/x-mathjax-config"">
      MathJax.Hub.Config({
        extensions: [""tex2jax.js""],
        jax: [""input/TeX"",""output/HTML-CSS""],
        tex2jax: {inlineMath: [[""$"",""$""],[""\\("",""\\)""]]}
      });
    </script>

<script type=""text/javascript""
  src=""http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"">
</script>
<style type=""text/css"">
h2 {
    page-break-before: always;
}
code {
    line-height: 150%;
}

.reserved{
    color : blue;
}
.string{
    color : red;
}
.comment{
    color : green;
}
.class{
    color : steelblue;
}
</style>

</head>
<body>
";

            foreach (string file_path in file_lines.Keys) {
                List<Token[]> lines = file_lines[file_path];

                StringWriter sw = new StringWriter();

                sw.WriteLine(header);

                sw.Write(Code(lines));
                sw.WriteLine("</body></html>");

                string home_dir = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)));

                string html_path = string.Format("{0}\\out\\{1}.html", home_dir, Path.GetFileNameWithoutExtension(file_path));

                File.WriteAllText(html_path, sw.ToString(), Encoding.UTF8);
            }
        }
    }

    public class Class {
        public string Name;
        public string SuperClassName = null;
        public List<Method> Methods = new List<Method>();

        public Class(string class_name) {
            Name = class_name;
        }
    }

    public class Method {
        public string Name;
        public List<Token[]> Lines = new List<Token[]>();

        public Method(string fnc_name) {
            Name = fnc_name;
        }
    }

}
