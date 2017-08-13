interface IEnumerable<T> {
}

class List<T> : IEnumerable<T> {
}
class Array<T> : IEnumerable<T> {
    int Length;
    int GetLength(int dimension);
}

interface IEnumerable<T> {
    bool Any();
    bool Contains(T x);
    int Count();
    IEnumerable<T> Distinct();
    T First();
    IEnumerable<T> Intersect(IEnumerable<T> v);
    T Last();
    IEnumerable<T> Skip(int count);
    IEnumerable<T> Take(int count);
    List<T> ToList();
    T[] ToArray();
    T Max();
    T Sum();
}

interface IEqualityComparer<T> {
}

class Action {
}
class Assembly {
}
class AssemblyName {
}
class Attribute {
}
class bool {
}
class byte {
}
class char {
}
class Color {
}
class Colors {
}
class CustomAttributeData {
}
class DateTime {
}
class Debug {
}
class Delegate {
}
class Dictionary<TKey, TValue> {
    public IEnumerable<TKey> Keys;
    public IEnumerable<TValue> Values;
    void Add(TKey key, TValue value);
    void Clear();
    bool ContainsKey(TKey key);
    public bool ContainsValue(TValue value);
    bool Remove(TKey key);
    bool TryGetValue(TKey key, out TValue value);
}
class Directory {
}
class double {
}
class Encoding {
}
class Enumerable<T> : IEnumerable<T> {
    //bool Any();
    //T First();
    //T[] ToArray();
    //int Count();
    T Max();
    T Sum();
}
class EventInfo {
}
class Exception {
}
class FieldInfo {
}
class File {
    string[] ReadAllLines(string path, Encoding encoding);
}
class float {
}
interface IEnumerator {
}
class int {
}
delegate bool Predicate<T>(T obj);
class List<T> {
    void Add(T item);
    void AddRange(IEnumerable<T> collection);
    void Clear();
    void CopyTo(int index, T[] array, int arrayIndex, int count);
    int Count;
    int FindIndex(Predicate<T> match);
    List<T> GetRange(int index, int count);
    int IndexOf(T item);
    void Insert(int index, T item);
    void InsertRange(int index, IEnumerable<T> collection);
    bool Remove(T item);
    void RemoveAt(int index);
    void RemoveRange(int index, int count);
    void Reverse();
    void Sort();
    //void Sort(Comparison<T> comparison);
    //T[] ToArray();
}
class ManualResetEvent {
}
class Math {
    public static double Sqrt(double d);
}
class MethodInfo {
}
class object {
}
class ParameterInfo {
}
class Path {
}
class Point {
}
class PropertyInfo {
}
class Rect {
    bool Contains(Point point);
}
class short {
}
class Size {
}
class Stack<T> : IEnumerable<T> {
    void Clear();
    int Count;
    T Peek();
    T Pop();
    void Push(T item);
}
class string {
}
class StringWriter {
}
class SynchronizationContext {
}
class Task {
}
class ThreadStatic {
}
class TimeSpan {
}
class Type {
    TypeInfo GetTypeInfo();
}
class TypeInfo {
}
class UnderlineType {
}
class UTF8Encoding {
}
class void {
}
class WebUtility {
}
