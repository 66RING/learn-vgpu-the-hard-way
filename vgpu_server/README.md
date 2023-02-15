# 设计

## 协议

⭐

同大多数流式传输, 我们需要先传输数据的大小然后再传输数据。为了兼容这个设计, 需要对结构体进行特别设计。首相如下结构体是不可行的因为它包含了一个指针, 我们传输时buf和matedata数据不连续，造成额外组织维护开销。

```c
typedef struct {
  int bufLen;
  uint8_t *buf;
} byte_t;
```

所以我们可以参考redis中sds的设计, 开辟空间时额外开辟用于存储matedata的空间, 用户再通过暴露的方法访问数据对象。

```c
// 用户不直接使用, 用于二进制安全的网络数据传递
typedef struct {
  int len;
  uint8_t data[]; // 不占用空间
} byteHdr;

typedef uint8_t* byte_t;
```

用户通过直接直接使用`byte_t`, 即`byteHdr`中的`data`部分, 然后根据我们的规则就可以访问他的元数据。具体数据用户可以直接当数组使用，
