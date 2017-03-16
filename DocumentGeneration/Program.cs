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
            string[] file_list = Directory.GetFiles(source_dir, "*.cs");

            StringWriter sw = new StringWriter();
            sw.WriteLine("@startuml");

            Dictionary<string, Class> class_dic = new Dictionary<string, Class>();

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


                foreach(Token[] tokens in lines) {

                    if(tokens.Length == 0) {
                        // 空行の場合

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

                        Class cls;

                        string class_name = tokens[idx + 1].Text;
                        if (!class_dic.TryGetValue(class_name, out cls)) {

                            cls = new Class(class_name);
                            class_dic.Add(class_name, cls);
                        }

                        if(idx + 3 < tokens.Length && tokens[idx + 2].Text == ":") {

                            cls.SuperClassName = tokens[idx + 3].Text;
                        }
                    }
                }
            }

            foreach(Class cls in class_dic.Values) {
                sw.WriteLine("class {0} {{", cls.Name);
                sw.WriteLine("}");
                if (cls.SuperClassName != null) {

                    sw.WriteLine("{0} <|-- {1}", cls.SuperClassName, cls.Name);
                }
            }

            sw.WriteLine("@enduml");
            File.WriteAllText("class.txt", sw.ToString(), Encoding.UTF8);
        }

        public int[] Range(int n) {
            return Enumerable.Range(0, n).ToArray();
        }
    }

    public class Class {
        public string Name;
        public string SuperClassName = null;

        public Class(string class_name) {
            Name = class_name;
        }
    }

}
