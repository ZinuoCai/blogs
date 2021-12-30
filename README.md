# Sharing Blogs Step by Step

1. 初始化项目文件夹

    ```
    hexo init blogs
    cd blogs
    npm install
    ```

2. 安装 Next 主题

    ```
    git clone https://github.com/ahonn/hexo-theme-even themes/even
    ```

    更改站点配置文件

    ```
    themes: next
    ```

4. 预览

    ```
    hexo g
    hexo s
    ```

5. 生成 Category 与 Tag

    ```
    hexo new page categories
    hexo new page tags
    ```

6. 图片解决办法

    - 在 `source` 目录下新建文件夹 `images`，将博客中的所有图片都移动到该目录下

    - 将文章中所有的图片路径修改为 `/images/your-image-name.png`


7. 公式解决方法

    https://ranmaosong.github.io/2017/11/29/hexo-support-mathjax/

8. 删除标题的编号

    站点配置文件中 `toc/number` 设置为 `false`


9. 部署

    ```
    npm install hexo-deployer-git --save
    ```

    站点配置文件：
    ```
    deploy
        type: 'git'
        repo: https://github.com/ZinuoCai/blogs.git
    ```
